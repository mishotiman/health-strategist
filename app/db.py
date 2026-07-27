"""Thin database access. Raw psycopg3 for now — visible SQL beats an ORM
while the schema is still moving.

Connections come from a shared pool: the previous connect-per-call opened a
fresh TCP + auth handshake for every query (several per request), which adds
real latency against a remote Postgres. The pool is built lazily on first use
so importing app modules needs no database (offline tests, eval scripts).

The agent's checkpointer keeps its own small pool (app/agent.py): it requires
autocommit + dict-row connections, while everything here relies on ordinary
transactional ones — multi-statement writes (e.g. promote_pending) depend on
commit-at-the-end semantics.
"""
import threading

from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1, max_size=10, open=True,
                    timeout=10,  # max wait for a free connection (and the cap
                                 # on how long "is the DB up?" checks block)
                    # revalidate pooled connections so one dropped by an idle
                    # timeout or a DB restart isn't handed to a request
                    check=ConnectionPool.check_connection,
                )
    return _pool


def get_connection():
    """A pooled connection, as a context manager:

        with get_connection() as conn, conn.cursor() as cur: ...

    On clean exit the pool commits and returns the connection; on an exception
    it rolls back — same contract callers already relied on with psycopg.
    """
    return _get_pool().connection()
