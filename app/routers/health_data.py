"""The user's own health data: normalized metric ingestion and the bloodwork
PDF upload manager."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import bloodwork
from app.db import get_connection
from app.deps import demo_user_id, readable_user, require_user
from app.ingestion import query_metrics, upsert_metrics

log = logging.getLogger(__name__)

router = APIRouter()


class MetricRecord(BaseModel):
    date: str            # YYYY-MM-DD
    metric_type: str     # canonical (see app.ingestion.CANONICAL_UNITS)
    value: float
    unit: str | None = None


class MetricsRequest(BaseModel):
    # No user_id: metrics always land on the authenticated caller's own account.
    source: str
    records: list[MetricRecord]


class RenameRequest(BaseModel):
    name: str


@router.post("/metrics")
def ingest_metrics(req: MetricsRequest, user_id: int = Depends(require_user)):
    """Ingest normalized metrics for the caller's own account.

    The user id comes from the session, never the request body — accepting it
    from the body previously let anyone write metrics into any account.
    """
    written = upsert_metrics(user_id, req.source, [r.model_dump() for r in req.records])
    return {"user_id": user_id, "source": req.source, "metrics_written": written}


@router.get("/metrics/{user_id}")
def get_metrics(user_id: int, type: str | None = None, limit: int = 100,
                session_user: int = Depends(readable_user)):
    # Only your own data (via login session) or the public sample account's.
    if user_id != session_user and user_id != demo_user_id():
        return JSONResponse(status_code=403,
                            content={"error": "not authorized to read this user's metrics"})
    return {"user_id": user_id, "metrics": query_metrics(user_id, type, limit)}


@router.post("/upload/bloodwork")
async def upload_bloodwork(file: UploadFile = File(...),
                           user_id: int = Depends(require_user)):
    """Upload a lab-report PDF; Claude classifies it and, if it's bloodwork,
    extracts values into normalized metrics for the current user."""
    is_pdf = (file.filename or "").lower().endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        return JSONResponse(status_code=400,
                            content={"status": "bad_file", "metrics_written": 0,
                                     "message": "Please upload a PDF lab report."})
    try:
        result = bloodwork.ingest_pdf(user_id, await file.read())
        if result.get("status") == "ok":
            name = os.path.splitext(file.filename or "")[0].strip() or "Bloodwork report"
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO bloodwork_documents (user_id, name, report_date, metrics_count) "
                    "VALUES (%s, %s, %s, %s) RETURNING id, uploaded_at",
                    (user_id, name, result.get("report_date"), result.get("metrics_written", 0)),
                )
                doc_id, uploaded_at = cur.fetchone()
                conn.commit()
            result["document"] = {"id": doc_id, "name": name,
                                  "report_date": result.get("report_date"),
                                  "metrics_count": result.get("metrics_written", 0),
                                  "uploaded_at": uploaded_at.isoformat()}
        return result
    except Exception:  # never surface a bare 500 to the uploader
        log.exception("bloodwork upload failed for user %s (file=%r)",
                      user_id, file.filename)
        return JSONResponse(status_code=200,
                            content={"status": "error", "metrics_written": 0,
                                     "message": "Something went wrong processing that file — "
                                                "please try again."})


@router.get("/bloodwork/documents")
def list_bloodwork_documents(user_id: int = Depends(readable_user)):
    """Bloodwork reports the current user has uploaded (for the upload manager)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, report_date, metrics_count, uploaded_at "
            "FROM bloodwork_documents WHERE user_id = %s ORDER BY uploaded_at DESC",
            (user_id,),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["report_date"] = r["report_date"].isoformat() if r["report_date"] else None
        r["uploaded_at"] = r["uploaded_at"].isoformat()
    return {"documents": rows}


@router.patch("/bloodwork/documents/{doc_id}")
def rename_bloodwork_document(doc_id: int, req: RenameRequest,
                              user_id: int = Depends(require_user)):
    name = req.name.strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "name cannot be empty"})
    with get_connection() as conn, conn.cursor() as cur:
        # scoped to the current user, so you can only rename your own documents
        cur.execute("UPDATE bloodwork_documents SET name = %s WHERE id = %s AND user_id = %s",
                    (name, doc_id, user_id))
        updated = cur.rowcount
        conn.commit()
    if not updated:
        return JSONResponse(status_code=404, content={"error": "document not found"})
    return {"ok": True, "id": doc_id, "name": name}


@router.delete("/bloodwork/documents/{doc_id}")
def delete_bloodwork_document(doc_id: int, user_id: int = Depends(require_user)):
    """Remove an uploaded report and the lab values it contributed. (Values are
    matched by report_date; a rare second report on the same date would share them.)"""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT report_date FROM bloodwork_documents WHERE id = %s AND user_id = %s",
                    (doc_id, user_id))
        row = cur.fetchone()
        if row is None:
            return JSONResponse(status_code=404, content={"error": "document not found"})
        report_date = row[0]
        cur.execute("DELETE FROM bloodwork_documents WHERE id = %s AND user_id = %s",
                    (doc_id, user_id))
        if report_date is not None:
            cur.execute("DELETE FROM health_metrics WHERE user_id = %s AND source = 'bloodwork' "
                        "AND metric_date = %s", (user_id, report_date))
        conn.commit()
    return {"ok": True, "id": doc_id}
