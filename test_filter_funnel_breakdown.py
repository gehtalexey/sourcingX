"""Unit tests for render_filter_funnel_breakdown.

Pins the parity contract between the Filter tab and Filter+ tab funnel
summaries: both tabs feed a {filter_name: int} counts dict into the same
helper, which must:
  - skip zero/None counts
  - render one bullet per non-zero count with the exact "**name**: N removed" shape
  - render the divider + heading + caption header regardless
  - emit the "No candidates were filtered out" info when everything is zero/empty
  - emit the restore hint when at least one filter dropped something

We use a tiny fake `st` module so we don't import Streamlit (the focused test
suite the project runs in CI doesn't have streamlit installed at import time
for dashboard.py; we test the helper in isolation by monkey-patching `st`).
"""

import importlib
import sys
import types
import unittest


class _FakeSt:
    """Minimal capture-only stand-in for streamlit."""

    def __init__(self):
        self.calls = []  # list of (method_name, args, kwargs)

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            # Return a tiny object that supports `with` for things like expander.
            class _Ctx:
                def __enter__(self_inner): return self_inner
                def __exit__(self_inner, *a): return False
            return _Ctx()
        return _record


def _load_helper():
    """Import dashboard.render_filter_funnel_breakdown without importing real streamlit.

    dashboard.py imports streamlit at module load. We don't want to pull all of
    streamlit just to test a pure rendering helper, so we install a fake `st`
    module before the import. We also stub a handful of other heavy modules
    dashboard.py may try to import at module level. If the import fails for
    reasons unrelated to our helper, we fall back to exec'ing only the helper's
    source so the unit test stays self-contained.
    """
    # Fast path: just exec the helper's source. This avoids importing the
    # 11k-line dashboard module (which pulls in streamlit, openai, gspread,
    # etc.) for a 20-line pure-render helper.
    import re
    with open('dashboard.py', 'r', encoding='utf-8') as f:
        src = f.read()
    # Grab the helper definition by name.
    pattern = r'(def render_filter_funnel_breakdown\(.*?\n)(?=\n\n@|\n\ndef |\Z)'
    m = re.search(pattern, src, re.DOTALL)
    if not m:
        raise RuntimeError("render_filter_funnel_breakdown not found in dashboard.py")
    helper_src = m.group(1)
    ns = {}
    fake_st = _FakeSt()
    ns['st'] = fake_st
    exec(helper_src, ns)
    return ns['render_filter_funnel_breakdown'], fake_st


class TestRenderFilterFunnelBreakdown(unittest.TestCase):
    def test_renders_one_bullet_per_nonzero_filter(self):
        helper, fake_st = _load_helper()
        counts = {
            "Past Candidates": 5,
            "Blacklist Companies": 0,
            "Excluded Titles": 12,
        }
        helper(counts)
        markdowns = [c for c in fake_st.calls if c[0] == "markdown"]
        # First markdown is the section heading; bullets follow.
        bullet_texts = [args[0] for _, args, _ in markdowns if args and args[0].startswith("- **")]
        self.assertEqual(len(bullet_texts), 2)
        self.assertIn("- **Past Candidates**: 5 removed", bullet_texts)
        self.assertIn("- **Excluded Titles**: 12 removed", bullet_texts)
        # Zero-count entries must not render.
        self.assertFalse(any("Blacklist Companies" in t for t in bullet_texts))

    def test_renders_restore_hint_when_anything_dropped(self):
        helper, fake_st = _load_helper()
        helper({"Past Candidates": 3})
        infos = [args[0] for name, args, _ in fake_st.calls if name == "info"]
        self.assertTrue(any("restore candidates" in i for i in infos))

    def test_renders_no_candidates_message_for_empty_or_zero(self):
        for counts in [{}, None, {"Past Candidates": 0, "Blacklist Companies": 0}]:
            helper, fake_st = _load_helper()
            helper(counts)
            infos = [args[0] for name, args, _ in fake_st.calls if name == "info"]
            self.assertTrue(any("No candidates were filtered out" in i for i in infos),
                            f"failed for counts={counts!r}")

    def test_section_header_always_rendered(self):
        helper, fake_st = _load_helper()
        helper({"Past Candidates": 1})
        markdowns = [args[0] for name, args, _ in fake_st.calls if name == "markdown"]
        self.assertIn("### Filtered Candidates Summary", markdowns)

    def test_skips_none_values(self):
        helper, fake_st = _load_helper()
        helper({"Past Candidates": None, "Excluded Titles": 4})
        markdowns = [args[0] for name, args, _ in fake_st.calls if name == "markdown"]
        bullets = [m for m in markdowns if m.startswith("- **")]
        self.assertEqual(bullets, ["- **Excluded Titles**: 4 removed"])


if __name__ == "__main__":
    unittest.main()
