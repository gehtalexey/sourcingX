"""Pure-Python helpers for matching company names against filter lists.

Import-safe: no streamlit / requests / openai imports. Both dashboard.py
and tests can import these directly without stubbing the world.

The two public helpers are:
  - normalize_company_name(name) -> str
  - company_matches_filter_list(company, company_list) -> bool

Matching rules (see company_matches_filter_list):
  1. Exact match after normalization ("Microsoft Israel" == "Microsoft").
  2. One name is a prefix of the other, both >=4 chars
     ("Bank Leumi" <-> "Bank Leumi Le-Israel").
  3. Token subset: the shorter name's tokens are a subset of the longer's
     tokens, AND the shorter has >=2 tokens. Catches the "Elad Software
     Systems" (list) vs "Elad Systems" (profile) case. The >=2-token guard
     stops a one-word entry like "Apple" from matching every "Apple X".
"""

try:
    import pandas as pd  # type: ignore

    def _isna(x):
        try:
            return bool(pd.isna(x))
        except (ValueError, TypeError):
            return x is None
except Exception:  # pragma: no cover - pandas always available in our env
    def _isna(x):
        return x is None


_COMPANY_NAME_SUFFIXES = (
    ' ltd', ' inc', ' corp', ' llc', ' limited', ' israel', ' il',
    ' technologies', ' tech', ' software', ' solutions', ' group',
)


def normalize_company_name(name) -> str:
    """Normalize a company name for filter matching.

    Lower-cases and strips a small set of legal / locale / generic suffixes
    that don't help identify the company (e.g. "Microsoft Israel" -> "microsoft").
    Returns '' for blank / NaN input.
    """
    if _isna(name) or not str(name).strip():
        return ''
    norm = str(name).lower().strip()
    for suffix in _COMPANY_NAME_SUFFIXES:
        if norm.endswith(suffix):
            norm = norm[:-len(suffix)].strip()
    return norm


def company_matches_filter_list(company, company_list) -> bool:
    """Return True if `company` matches any entry in `company_list`.

    Matches in this order:
      1. Exact after normalization ("Microsoft Israel" == "Microsoft").
      2. One name is a prefix of the other, both >=4 chars ("Bank Leumi" <->
         "Bank Leumi Le-Israel").
      3. Token subset: the shorter name's tokens are a subset of the longer's
         tokens, AND the shorter has >=2 tokens. Catches the "Elad Software
         Systems" (list) vs "Elad Systems" (profile) case. The >=2-token guard
         stops a one-word entry like "Apple" from matching every "Apple X".
    """
    if _isna(company) or not str(company).strip():
        return False
    company_norm = normalize_company_name(company)
    if not company_norm:
        return False
    company_tokens = set(company_norm.split())
    for c in company_list:
        c_norm = normalize_company_name(c)
        if not c_norm:
            continue
        if company_norm == c_norm:
            return True
        if len(c_norm) >= 4 and len(company_norm) >= 4:
            if company_norm.startswith(c_norm) or c_norm.startswith(company_norm):
                return True
        c_tokens = set(c_norm.split())
        if c_tokens and company_tokens:
            if len(c_tokens) <= len(company_tokens):
                shorter, longer = c_tokens, company_tokens
            else:
                shorter, longer = company_tokens, c_tokens
            if len(shorter) >= 2 and shorter.issubset(longer):
                return True
    return False
