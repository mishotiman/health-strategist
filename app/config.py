"""App settings, loaded from environment (.env in dev, real env in prod).

The single source of runtime configuration: app modules read `settings.X`
rather than os.environ directly, so .env behaves identically everywhere
(pydantic loads it here; a bare os.environ.get would miss it outside
docker-compose) and every knob is discoverable in one place. Standalone
scripts (evals, corpus tooling) may still read their own env vars.
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://phs:phs@db:5432/phs"

    # Public origin of the app — cookie hardening (Secure over https), the links
    # in verification/reset emails, and the OAuth redirect URIs.
    app_base_url: str = "http://localhost:8000"

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

    # Generation models. The flagship agent (/chat) runs Opus; the RAG pipeline
    # (/ask + evals) runs Sonnet at near-Opus quality for ~1/2 the cost. Point
    # AGENT_MODEL at claude-haiku-4-5 for cheap agent-eval smoke runs.
    agent_model: str = "claude-opus-4-8"
    rag_gen_model: str = "claude-sonnet-5"

    # Rate limits — requests allowed per window on the endpoints that spend on
    # the app's API keys (windows are fixed in app/ratelimit.py).
    rate_chat_user: int = 20
    rate_chat_guest: int = 8
    rate_ask_ip: int = 10
    rate_search_ip: int = 30

    whoop_client_id: str = ""
    whoop_client_secret: str = ""
    whoop_redirect_uri: str = "http://localhost:8000/whoop/callback"

    # Google / Microsoft sign-in (OIDC). "common" lets both personal and
    # work/school Microsoft accounts in.
    google_client_id: str = ""
    google_client_secret: str = ""
    microsoft_client_id: str = ""
    microsoft_client_secret: str = ""
    microsoft_tenant: str = "common"

    # Transactional email (verification + password reset).
    resend_api_key: str = ""
    email_from: str = "Personal Health Strategist <onboarding@resend.dev>"

    @field_validator("app_base_url")
    @classmethod
    def _no_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


settings = Settings()
