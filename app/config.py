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

    # Encryption at rest for provider OAuth tokens (app/crypto.py). Unset means
    # tokens are stored in plaintext, which is the pre-existing behaviour and
    # what dev/CI run with; set it in production. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    token_encryption_key: str = ""

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

    # Reranking (app/rag.py). Bi-encoder cosine ranks by "is this in the same
    # region of meaning-space"; a reranker reads query and passage TOGETHER and
    # scores actual relevance, which is what fixes ordering. Off by default: it
    # adds an API call and per-token cost to every retrieval, so turn it on only
    # where the MRR gain is worth it. When on, retrieval fetches rag_fetch_k
    # candidates and the reranker reorders them down to k.
    rag_rerank: bool = False
    rag_fetch_k: int = 50
    rerank_model: str = "rerank-2.5"

    # Evidence diversity: at most this many passages from any one paper in the
    # top-k. Chunks from the same paper cluster in meaning-space, so an uncapped
    # top-6 is routinely dominated by one document — measured over the golden
    # set it drew on a mean of 3.25 distinct papers, one question taking all six
    # from a SINGLE paper. For a product whose claim is breadth of peer-reviewed
    # grounding, that is worth fixing.
    #
    # But it is a trade, not a free win, and the full eval priced it (diversity /
    # correctness, 24 and 31 cases):
    #     cap off  3.25 papers (min 1, seven questions <=2)   correctness 0.97
    #     cap 3    3.62 papers (min 2, one question <=2)      correctness 0.97
    #     cap 2    4.29 papers (min 3, none <=2)              correctness 0.84
    # Capping to 2 swaps the authoritative paper's 3rd-5th passages — the ones
    # carrying the specific numbers — for weaker passages elsewhere, and answer
    # correctness pays for the extra citations. 3 takes the diversity that is
    # free and stops at the knee. 0 disables the cap.
    rag_max_chunks_per_doc: int = 3

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
