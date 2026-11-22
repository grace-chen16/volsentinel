import os
import sys
import tempfile
from pathlib import Path

import pytest

# isolate test runs from the real runs/ and memory, and skip the (slow) news archive
_TMP = Path(tempfile.mkdtemp(prefix="volsentinel-tests-"))
os.environ.setdefault("VOLSENTINEL_RUNS", str(_TMP / "runs"))
os.environ.setdefault("VOLSENTINEL_MEMORY", str(_TMP / "runs" / "experience.db"))
os.environ["VOLSENTINEL_LOAD_NEWS"] = "0"
os.environ["VOLSENTINEL_LLM"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volsentinel.config import Settings  # noqa: E402
from volsentinel.context import Context  # noqa: E402


@pytest.fixture(scope="session")
def ctx():
    s = Settings()
    if not s.db_path.exists():
        pytest.skip(f"project database not found at {s.db_path}")
    return Context(s, load_news=False)


@pytest.fixture
def fresh_ctx(ctx, tmp_path):
    """Same data, but an empty memory and run store, so tests do not see each other."""
    from volsentinel.memory import ExperienceMemory
    from volsentinel.state import StateStore
    old = ctx.memory, ctx.store
    ctx.memory, ctx.store = ExperienceMemory(tmp_path / "mem.db"), StateStore(tmp_path / "runs")
    yield ctx
    ctx.memory, ctx.store = old
