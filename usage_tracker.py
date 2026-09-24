"""
Usage Tracker Module for LinkedIn Enricher
Tracks API consumption across all providers: Crustdata, PhantomBuster, SalesQL, OpenAI
"""

import time
import warnings
from datetime import datetime
from typing import Optional
from functools import wraps


# Crustdata pricing: $1,500 USD for 150,000 credits (50,000 profiles at 3 credits each)
CRUSTDATA_PRICING = {
    'total_credits': 150_000,
    'total_cost_usd': 1500.00,
    'cost_per_credit': 0.01,       # $1,500 / 150,000 = $0.01 per credit
    'cost_per_profile': 0.03,      # $0.01 * 3 credits = $0.03 per profile
    'credits_per_profile': 3,
}

# New v2025-11-01 batch-enrich pricing (POST /batch/person/enrich): additive,
# base profile = 1 credit (vs. 3 credits/profile for the legacy enrich
# endpoint above). Verified against docs.crustdata.com 2026-07-20.
CRUSTDATA_PRICING_V2_ENRICH = {
    'credits_per_profile_base': 1,
}

# OpenAI pricing (per 1M tokens)
OPENAI_PRICING = {
    'gpt-5.6-luna': {
        'input': 0.20,        # $0.20 per 1M input tokens
        'output': 1.20,       # $1.20 per 1M output tokens
        'cached_input': 0.02, # $0.02 per 1M cached-prefix input tokens
    },
    'gpt-4.1-mini': {
        'input': 0.40,   # $0.40 per 1M input tokens
        'output': 1.60,  # $1.60 per 1M output tokens
    },
    'gpt-4o-mini': {
        'input': 0.15,   # $0.15 per 1M input tokens
        'output': 0.60,  # $0.60 per 1M output tokens
    },
    'gpt-4o': {
        'input': 2.50,   # $2.50 per 1M input tokens
        'output': 10.00, # $10.00 per 1M output tokens
    }
}


def _openai_pricing_for(model: str) -> dict:
    """Look up OPENAI_PRICING for `model`. An unrecognized model previously
    fell back to gpt-4o-mini's rate silently -- underreporting cost by up to
    50% (the exact gap Codex found in PR #131 for gpt-5.6-luna itself, before
    that entry was added). Now warns loudly instead, so a new/renamed model
    (e.g. wiring Jev in without a pricing entry) can't hide a cost bug."""
    pricing = OPENAI_PRICING.get(model)
    if pricing is None:
        warnings.warn(
            f"No OpenAI pricing entry for model {model!r} -- falling back to "
            "gpt-4o-mini rates, which is almost certainly wrong. Add a real "
            "entry to OPENAI_PRICING in usage_tracker.py.",
            stacklevel=3,
        )
        pricing = OPENAI_PRICING['gpt-4o-mini']
    return pricing


class UsageTracker:
    """Tracks and logs API usage to Supabase."""

    def __init__(self, db_client=None):
        """Initialize tracker with optional database client.

        Args:
            db_client: SupabaseClient instance for logging to database
        """
        self.db_client = db_client

    def log_usage(
        self,
        provider: str,
        operation: str,
        request_count: int = 1,
        credits_used: float = None,
        tokens_input: int = None,
        tokens_output: int = None,
        cost_usd: float = None,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: int = None,
        metadata: dict = None
    ) -> Optional[dict]:
        """Log an API usage event.

        Args:
            provider: API provider name (crustdata, phantombuster, salesql, openai)
            operation: Operation type (enrich, scrape, email_lookup, screen, etc.)
            request_count: Number of API requests made
            credits_used: Credits/lookups consumed (for credit-based APIs)
            tokens_input: Input tokens (for OpenAI)
            tokens_output: Output tokens (for OpenAI)
            cost_usd: Calculated cost in USD
            status: 'success' or 'error'
            error_message: Error details if status is 'error'
            response_time_ms: API response time in milliseconds
            metadata: Additional JSON metadata

        Returns:
            The inserted record or None if no db_client
        """
        if not self.db_client:
            return None

        data = {
            'provider': provider.lower(),
            'operation': operation,
            'request_count': request_count,
            'status': status,
            'created_at': datetime.utcnow().isoformat(),
        }

        if credits_used is not None:
            data['credits_used'] = credits_used
        if tokens_input is not None:
            data['tokens_input'] = tokens_input
        if tokens_output is not None:
            data['tokens_output'] = tokens_output
        if cost_usd is not None:
            data['cost_usd'] = cost_usd
        if error_message:
            data['error_message'] = error_message
        if response_time_ms is not None:
            data['response_time_ms'] = response_time_ms
        if metadata:
            data['metadata'] = metadata

        try:
            result = self.db_client.insert('api_usage_logs', data)
            return result[0] if result else None
        except Exception as e:
            # Don't let logging failures break the app
            print(f"[UsageTracker] Failed to log usage: {e}")
            return None

    def log_crustdata(
        self,
        profiles_enriched: int,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: int = None
    ) -> Optional[dict]:
        """Log Crustdata enrichment usage.

        Crustdata charges 3 credits per profile enriched.
        Pricing: $1,500 for 150,000 credits ($0.01/credit, $0.03/profile).
        """
        credits = profiles_enriched * CRUSTDATA_PRICING['credits_per_profile']
        cost_usd = credits * CRUSTDATA_PRICING['cost_per_credit']
        return self.log_usage(
            provider='crustdata',
            operation='enrich',
            request_count=1,
            credits_used=credits,
            cost_usd=cost_usd,
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms,
            metadata={'profiles_enriched': profiles_enriched}
        )

    def log_crustdata_batch_enrich(
        self,
        requested: int,
        fulfilled: int,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: int = None,
    ) -> Optional[dict]:
        """Log usage for the new v2025-11-01 batch-enrich endpoint
        (POST /batch/person/enrich) — additive pricing, base profile = 1
        credit, distinct from log_crustdata()'s 3-credits/profile legacy
        rate. Do NOT reuse log_crustdata() for this; it hardcodes the wrong
        per-profile credit cost.

        Billed on `fulfilled`, not `requested` — Crustdata's global no-charge-
        on-no-match policy means unmatched profiles cost nothing.
        """
        credits = fulfilled * CRUSTDATA_PRICING_V2_ENRICH['credits_per_profile_base']
        cost_usd = credits * CRUSTDATA_PRICING['cost_per_credit']
        return self.log_usage(
            provider='crustdata',
            operation='batch_enrich',
            request_count=1,
            credits_used=credits,
            cost_usd=cost_usd,
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms,
            metadata={'requested': requested, 'fulfilled': fulfilled, 'unmatched': requested - fulfilled},
        )

    def log_crustdata_sync_enrich(
        self,
        requested: int,
        fulfilled: int,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: int = None,
    ) -> Optional[dict]:
        """Log usage for the synchronous v2025-11-01 enrich endpoint
        (POST /person/enrich) — same additive pricing as
        log_crustdata_batch_enrich() (base profile = 1 credit), distinct
        operation name since this is a single inline call, not an async
        batch job. Do NOT reuse log_crustdata() for this; it hardcodes the
        wrong (3 credits/profile, legacy-endpoint) per-profile cost.

        Billed on `fulfilled`, not `requested` — Crustdata's global no-charge-
        on-no-match policy means an unmatched profile costs nothing.
        """
        credits = fulfilled * CRUSTDATA_PRICING_V2_ENRICH['credits_per_profile_base']
        cost_usd = credits * CRUSTDATA_PRICING['cost_per_credit']
        return self.log_usage(
            provider='crustdata',
            operation='sync_enrich',
            request_count=1,
            credits_used=credits,
            cost_usd=cost_usd,
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms,
            metadata={'requested': requested, 'fulfilled': fulfilled, 'unmatched': requested - fulfilled},
        )

    def log_salesql(
        self,
        lookups: int = 1,
        emails_found: int = 0,
        status: str = None,
        error_message: str = None,
        response_time_ms: int = None,
        billed: bool = None,
    ) -> Optional[dict]:
        """Log SalesQL email lookup usage.

        Billing rule (salesql-api skill): SalesQL charges a credit only for a
        lookup that returns at least one email or phone. No-result lookups
        are free, and with match_if_direct_email=true only a Direct email is
        charged. So credits_used = lookups when billed, else 0. The row is
        still written for a miss (status 'not_found') so the hit rate stays
        auditable. `billed` defaults to "an email came back".
        """
        if billed is None:
            billed = emails_found > 0
        if status is None:
            status = 'success' if billed else 'not_found'
        return self.log_usage(
            provider='salesql',
            operation='email_lookup',
            request_count=lookups,
            credits_used=lookups if billed else 0,
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms,
            metadata={'emails_found': emails_found}
        )

    def log_openai(
        self,
        tokens_input: int,
        tokens_output: int,
        model: str = 'gpt-4o-mini',
        profiles_screened: int = 1,
        status: str = 'success',
        error_message: str = None,
        response_time_ms: int = None,
        use_flex: bool = False,
        cached_tokens: int = 0,
        request_count: int = 1
    ) -> Optional[dict]:
        """Log OpenAI API usage with cost calculation.

        Args:
            tokens_input: Number of input tokens (includes any cached_tokens)
            tokens_output: Number of output tokens
            model: Model name for pricing lookup
            profiles_screened: Number of profiles screened in this call
            use_flex: Whether this request was billed at OpenAI's flex-tier
                rate (half the standard price)
            cached_tokens: Portion of tokens_input served from OpenAI's
                prompt cache, billed at the model's discounted cached rate
                when known (falls back to the full input rate otherwise)
            request_count: Number of API requests this log entry represents
                (e.g. 2 when an empty-response retry fired)
        """
        # Calculate cost
        pricing = _openai_pricing_for(model)
        cached_tokens = min(cached_tokens, tokens_input)
        uncached_input_tokens = tokens_input - cached_tokens
        cached_rate = pricing.get('cached_input', pricing['input'])
        cost_usd = (
            (uncached_input_tokens / 1_000_000) * pricing['input'] +
            (cached_tokens / 1_000_000) * cached_rate +
            (tokens_output / 1_000_000) * pricing['output']
        )
        if use_flex:
            cost_usd /= 2

        return self.log_usage(
            provider='openai',
            operation='screen',
            request_count=request_count,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms,
            metadata={
                'model': model,
                'profiles_screened': profiles_screened,
                'service_tier': 'flex' if use_flex else 'standard',
                'cached_tokens': cached_tokens
            }
        )

    def log_phantombuster(
        self,
        operation: str = 'scrape',
        profiles_scraped: int = 0,
        status: str = 'success',
        error_message: str = None,
        agent_id: str = None,
        container_id: str = None
    ) -> Optional[dict]:
        """Log PhantomBuster usage.

        PhantomBuster uses credits based on execution time and phantom type.
        """
        return self.log_usage(
            provider='phantombuster',
            operation=operation,
            request_count=1,
            credits_used=1,  # 1 run = 1 credit (simplified)
            status=status,
            error_message=error_message,
            metadata={
                'profiles_scraped': profiles_scraped,
                'agent_id': agent_id,
                'container_id': container_id
            }
        )


def calculate_openai_cost(tokens_input: int, tokens_output: int, model: str = 'gpt-4o-mini') -> float:
    """Calculate OpenAI API cost in USD.

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        model: Model name

    Returns:
        Cost in USD
    """
    pricing = _openai_pricing_for(model)
    return (
        (tokens_input / 1_000_000) * pricing['input'] +
        (tokens_output / 1_000_000) * pricing['output']
    )


# Fallback AI cost per screened candidate when there is no logged history for
# the model yet. Screening makes 2 model calls per candidate (verdict + bonus
# pass); ~$0.003 per candidate is the observed figure for gpt-5.6-luna
# (the old token-based estimate was ~4x too low). Once a run is logged, the
# real per-candidate cost from api_usage_logs replaces this.
DEFAULT_SCREEN_COST_PER_CANDIDATE = 0.003

# Two logged screening calls further apart than this belong to different runs.
SCREEN_RUN_GAP_MINUTES = 10


def cost_per_candidate_from_logs(rows: list, gap_minutes: int = SCREEN_RUN_GAP_MINUTES) -> Optional[float]:
    """Real AI cost per candidate of the most recent screening run.

    `rows` are api_usage_logs rows for one screening model (operation
    'screen', any order). The latest run = the newest row plus every row
    before it with no gap longer than `gap_minutes` between neighbours.
    Cost per candidate (standard rate) = sum(cost_usd) / sum(metadata.profiles_screened);
    the bonus-pass call logs profiles_screened=0, so its tokens count toward
    cost without double-counting the candidate. Returns None when there is
    nothing usable (no rows, no candidates, no cost)."""
    parsed = []
    for r in rows or []:
        try:
            ts = datetime.fromisoformat(str(r.get('created_at', '')).replace('Z', '+00:00'))
        except ValueError:
            continue
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        parsed.append((ts, r))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0], reverse=True)

    run = [parsed[0][1]]
    for (newer_ts, _), (ts, r) in zip(parsed, parsed[1:]):
        if (newer_ts - ts).total_seconds() > gap_minutes * 60:
            break
        run.append(r)

    # Normalise to the standard (non-flex) rate: a flex row was billed at
    # half price, so double it back. The caller halves again when the next
    # run is on flex.
    cost = sum(
        float(r.get('cost_usd') or 0) * (2 if (r.get('metadata') or {}).get('service_tier') == 'flex' else 1)
        for r in run
    )
    candidates = sum(int((r.get('metadata') or {}).get('profiles_screened') or 0) for r in run)
    if candidates <= 0 or cost <= 0:
        return None
    return cost / candidates


def recent_screening_cost_per_candidate(db_client, model: str, limit: int = 1000) -> Optional[float]:
    """Read recent api_usage_logs screening rows for `model` and return the
    last run's real cost per candidate (see cost_per_candidate_from_logs).
    Returns None on no history or any read failure."""
    if not db_client or not model:
        return None
    try:
        rows = db_client.select(
            'api_usage_logs',
            'created_at,cost_usd,metadata',
            {
                'operation': 'eq.screen',
                'status': 'eq.success',
                'metadata->>model': f'eq.{model}',
            },
            limit=limit,
            order_by='created_at.desc',
        )
    except Exception as e:
        print(f"[UsageTracker] Could not read screening history: {e}")
        return None
    return cost_per_candidate_from_logs(rows)


def estimate_screening_cost(n_candidates: int, n_thin: int,
                            recent_cost_per_candidate: Optional[float] = None) -> dict:
    """What an AI Screen click will cost, before it runs.

    n_candidates: profiles that will be screened.
    n_thin: of those, profiles missing both skills and summary (and not on
        the re-enrich cooldown) -- each gets a 1-credit Crustdata top-up.
    recent_cost_per_candidate: last run's real AI $/candidate from the logs;
        None falls back to DEFAULT_SCREEN_COST_PER_CANDIDATE.

    Returns {crustdata_credits, crustdata_usd, ai_usd, ai_cost_per_candidate,
    from_history}."""
    n_candidates = max(0, int(n_candidates or 0))
    n_thin = max(0, min(int(n_thin or 0), n_candidates))
    from_history = bool(recent_cost_per_candidate and recent_cost_per_candidate > 0)
    per_candidate = recent_cost_per_candidate if from_history else DEFAULT_SCREEN_COST_PER_CANDIDATE
    credits = n_thin * CRUSTDATA_PRICING_V2_ENRICH['credits_per_profile_base']
    return {
        'crustdata_credits': credits,
        'crustdata_usd': credits * CRUSTDATA_PRICING['cost_per_credit'],
        'ai_usd': n_candidates * per_candidate,
        'ai_cost_per_candidate': per_candidate,
        'from_history': from_history,
    }


def track_api_call(tracker: UsageTracker, provider: str, operation: str):
    """Decorator to track API calls with timing.

    Usage:
        @track_api_call(tracker, 'crustdata', 'enrich')
        def my_api_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                tracker.log_usage(
                    provider=provider,
                    operation=operation,
                    response_time_ms=elapsed_ms,
                    status='success'
                )
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                tracker.log_usage(
                    provider=provider,
                    operation=operation,
                    response_time_ms=elapsed_ms,
                    status='error',
                    error_message=str(e)[:500]
                )
                raise
        return wrapper
    return decorator
