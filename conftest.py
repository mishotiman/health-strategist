"""Pytest bootstrap.

Some app modules construct API clients at import time (anthropic, voyage). The
unit tests here only exercise PURE functions and never make a network call, so
we set placeholder keys before collection to let those imports succeed offline.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
