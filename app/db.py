"""Thin database access. Raw psycopg3 for now — visible SQL beats an ORM
while the schema is still moving."""
import psycopg

from app.config import settings


def get_connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)
