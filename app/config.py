"""App settings, loaded from environment (.env in dev, real env in prod)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://phs:phs@db:5432/phs"

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    # Note: OpenAI is a corpus-embedding fallback used directly by
    # scripts/embed_chunks.py (reads OPENAI_API_KEY from the environment). It is
    # intentionally not surfaced here because nothing at runtime reads it.

    # Embeddings — the model and dimensions MUST be identical on the corpus side
    # (scripts/embed_chunks.py) and the query side (app/rag.py), or retrieval
    # silently returns garbage. Change both at once by editing these two values.
    # voyage-3.5 supports Matryoshka dims (256/512/1024); 512 halves storage/RAM
    # vs 1024 with negligible quality loss.
    embedding_model: str = "voyage-3.5"
    embedding_dim: int = 512

    whoop_client_id: str = ""
    whoop_client_secret: str = ""


settings = Settings()
