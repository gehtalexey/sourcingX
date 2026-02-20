"""
API Design Helpers for SourcingX

Implements standardized API patterns:
1. PaginatedResponse - Consistent pagination model with page info
2. APIErrorResponse - Consistent error response format
3. RateLimiter - Rate limiting helpers for external API calls

These patterns ensure consistent API interactions throughout the application.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import TypeVar, Generic, List, Optional, Any, Dict, Callable
from datetime import datetime
from functools import wraps
from enum import Enum


# =============================================================================
# 1. PAGINATION RESPONSE MODELS
# =============================================================================

T = TypeVar('T')


@dataclass
class PaginationInfo:
    """Metadata about the current page of results."""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def from_offset(cls, offset: int, limit: int, total: int) -> 'PaginationInfo':
        """Create pagination info from offset-based pagination."""
        page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = (total + limit - 1) // limit if limit > 0 else 1
        return cls(
            page=page,
            page_size=limit,
            total_items=total,
            total_pages=total_pages,
            has_next=offset + limit < total,
            has_previous=offset > 0
        )


@dataclass
class PaginatedResponse(Generic[T]):
    """
    Standard paginated response wrapper.

    Usage:
        profiles = db.select_paginated('profiles', page=2, page_size=50)
        response = PaginatedResponse.create(
            items=profiles['data'],
            total=profiles['total'],
            page=2,
            page_size=50
        )

        # Access data
        for item in response.items:
            process(item)

        # Check pagination
        if response.pagination.has_next:
            fetch_next_page()
    """
    items: List[T]
    pagination: PaginationInfo

    @classmethod
    def create(cls, items: List[T], total: int, page: int = 1, page_size: int = 20) -> 'PaginatedResponse[T]':
        """Create a paginated response from items and metadata."""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )
        return cls(items=items, pagination=pagination)

    @classmethod
    def empty(cls, page_size: int = 20) -> 'PaginatedResponse[T]':
        """Create an empty paginated response."""
        return cls.create(items=[], total=0, page=1, page_size=page_size)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'items': self.items,
            'pagination': {
                'page': self.pagination.page,
                'page_size': self.pagination.page_size,
                'total_items': self.pagination.total_items,
                'total_pages': self.pagination.total_pages,
                'has_next': self.pagination.has_next,
                'has_previous': self.pagination.has_previous,
            }
        }


# =============================================================================
# 2. CONSISTENT ERROR RESPONSE FORMAT
# =============================================================================

class ErrorCode(str, Enum):
    """Standard error codes for API responses."""
    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    INVALID_INPUT = "INVALID_INPUT"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


@dataclass
class ErrorDetail:
    """Detailed information about a specific error."""
    field: Optional[str]
    message: str
    code: str

    def to_dict(self) -> Dict[str, Any]:
        result = {'message': self.message, 'code': self.code}
        if self.field:
            result['field'] = self.field
        return result


@dataclass
class APIErrorResponse:
    """
    Standard error response format for all API operations.

    Usage:
        try:
            result = api_call()
        except requests.HTTPError as e:
            error = APIErrorResponse.from_http_error(e)
            log_error(error)
            return error.to_dict()

        # For validation errors
        error = APIErrorResponse.validation_error(
            "Invalid input",
            details=[
                ErrorDetail(field="email", message="Invalid email format", code="INVALID_FORMAT"),
                ErrorDetail(field="name", message="Name is required", code="REQUIRED"),
            ]
        )
    """
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None

    @classmethod
    def from_exception(cls, e: Exception, code: ErrorCode = ErrorCode.INTERNAL_ERROR) -> 'APIErrorResponse':
        """Create error response from an exception."""
        return cls(
            error_code=code.value,
            message=str(e)[:500],  # Truncate long error messages
        )

    @classmethod
    def from_http_error(cls, response, provider: str = None) -> 'APIErrorResponse':
        """Create error response from an HTTP response."""
        status_code = response.status_code if hasattr(response, 'status_code') else 500

        # Map HTTP status codes to error codes
        code_map = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            409: ErrorCode.DUPLICATE_ENTRY,
            429: ErrorCode.RATE_LIMITED,
            500: ErrorCode.INTERNAL_ERROR,
            502: ErrorCode.EXTERNAL_API_ERROR,
            503: ErrorCode.SERVICE_UNAVAILABLE,
            504: ErrorCode.TIMEOUT_ERROR,
        }
        error_code = code_map.get(status_code, ErrorCode.EXTERNAL_API_ERROR)

        # Try to extract error message from response
        message = f"HTTP {status_code}"
        try:
            if hasattr(response, 'json'):
                data = response.json()
                if isinstance(data, dict):
                    message = data.get('error') or data.get('message') or data.get('detail') or message
            elif hasattr(response, 'text'):
                message = response.text[:200] if response.text else message
        except Exception:
            pass

        if provider:
            message = f"[{provider}] {message}"

        return cls(error_code=error_code.value, message=message)

    @classmethod
    def validation_error(cls, message: str, details: List[ErrorDetail] = None) -> 'APIErrorResponse':
        """Create a validation error response."""
        return cls(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message=message,
            details=details
        )

    @classmethod
    def not_found(cls, resource: str, identifier: Any = None) -> 'APIErrorResponse':
        """Create a not found error response."""
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        return cls(error_code=ErrorCode.NOT_FOUND.value, message=message)

    @classmethod
    def rate_limited(cls, provider: str, retry_after: int = None) -> 'APIErrorResponse':
        """Create a rate limit error response."""
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        return cls(error_code=ErrorCode.RATE_LIMITED.value, message=message)

    @classmethod
    def external_api_error(cls, provider: str, message: str) -> 'APIErrorResponse':
        """Create an external API error response."""
        return cls(
            error_code=ErrorCode.EXTERNAL_API_ERROR.value,
            message=f"[{provider}] {message}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'error_code': self.error_code,
            'message': self.message,
            'timestamp': self.timestamp,
        }
        if self.details:
            result['details'] = [d.to_dict() for d in self.details]
        if self.request_id:
            result['request_id'] = self.request_id
        return result

    def is_retryable(self) -> bool:
        """Check if this error is potentially retryable."""
        retryable_codes = {
            ErrorCode.RATE_LIMITED.value,
            ErrorCode.SERVICE_UNAVAILABLE.value,
            ErrorCode.TIMEOUT_ERROR.value,
            ErrorCode.EXTERNAL_API_ERROR.value,
        }
        return self.error_code in retryable_codes


# =============================================================================
# 3. RATE LIMITING HELPERS FOR EXTERNAL API CALLS
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int
    requests_per_day: Optional[int] = None
    burst_limit: Optional[int] = None  # Max concurrent requests

    @classmethod
    def for_crustdata(cls) -> 'RateLimitConfig':
        """Rate limit config for Crustdata API."""
        return cls(requests_per_minute=60, burst_limit=10)

    @classmethod
    def for_salesql(cls) -> 'RateLimitConfig':
        """Rate limit config for SalesQL API (180/min, 5000/day)."""
        return cls(requests_per_minute=140, requests_per_day=5000, burst_limit=20)

    @classmethod
    def for_phantombuster(cls) -> 'RateLimitConfig':
        """Rate limit config for PhantomBuster API."""
        return cls(requests_per_minute=30, burst_limit=5)

    @classmethod
    def for_openai(cls) -> 'RateLimitConfig':
        """Rate limit config for OpenAI API."""
        return cls(requests_per_minute=500, burst_limit=50)


class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Usage:
        # Create limiter for SalesQL
        limiter = RateLimiter(RateLimitConfig.for_salesql())

        # In your API call function
        def call_salesql(url):
            limiter.wait_if_needed()  # Blocks if rate limit reached
            response = requests.get(url)
            limiter.record_request()
            return response

        # Or use the decorator
        @limiter.limit
        def call_salesql(url):
            return requests.get(url)
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.Lock()
        self._minute_timestamps: List[float] = []
        self._day_timestamps: List[float] = []
        self._active_requests = 0

    def _prune_old_timestamps(self, now: float):
        """Remove timestamps outside the sliding window."""
        minute_cutoff = now - 60.0
        day_cutoff = now - 86400.0

        self._minute_timestamps = [t for t in self._minute_timestamps if t > minute_cutoff]
        if self.config.requests_per_day:
            self._day_timestamps = [t for t in self._day_timestamps if t > day_cutoff]

    def can_proceed(self) -> bool:
        """Check if a request can proceed without waiting."""
        with self._lock:
            now = time.time()
            self._prune_old_timestamps(now)

            # Check burst limit
            if self.config.burst_limit and self._active_requests >= self.config.burst_limit:
                return False

            # Check per-minute limit
            if len(self._minute_timestamps) >= self.config.requests_per_minute:
                return False

            # Check per-day limit
            if self.config.requests_per_day:
                if len(self._day_timestamps) >= self.config.requests_per_day:
                    return False

            return True

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit is reached. Returns time waited in seconds.

        This method blocks until a request slot is available.
        """
        total_wait = 0.0

        while True:
            with self._lock:
                now = time.time()
                self._prune_old_timestamps(now)

                # Check burst limit
                if self.config.burst_limit and self._active_requests >= self.config.burst_limit:
                    wait_time = 0.1  # Small wait for concurrent requests to complete
                elif len(self._minute_timestamps) >= self.config.requests_per_minute:
                    # Wait until oldest request in window expires
                    oldest = self._minute_timestamps[0] if self._minute_timestamps else now
                    wait_time = max(0.1, oldest + 60.0 - now)
                elif self.config.requests_per_day and len(self._day_timestamps) >= self.config.requests_per_day:
                    # Daily limit hit - this is a hard stop
                    raise RateLimitExceeded(
                        f"Daily rate limit of {self.config.requests_per_day} requests exceeded"
                    )
                else:
                    # Can proceed
                    self._active_requests += 1
                    return total_wait

            # Wait outside the lock
            time.sleep(wait_time)
            total_wait += wait_time

    def record_request(self, completed: bool = True):
        """Record that a request was made (or completed)."""
        with self._lock:
            now = time.time()
            if completed:
                self._active_requests = max(0, self._active_requests - 1)
            self._minute_timestamps.append(now)
            if self.config.requests_per_day:
                self._day_timestamps.append(now)

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        with self._lock:
            now = time.time()
            self._prune_old_timestamps(now)
            return {
                'requests_last_minute': len(self._minute_timestamps),
                'requests_limit_minute': self.config.requests_per_minute,
                'requests_last_day': len(self._day_timestamps) if self.config.requests_per_day else None,
                'requests_limit_day': self.config.requests_per_day,
                'active_requests': self._active_requests,
                'burst_limit': self.config.burst_limit,
            }

    def limit(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            try:
                return func(*args, **kwargs)
            finally:
                self.record_request()
        return wrapper


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


# =============================================================================
# GLOBAL RATE LIMITERS (Singleton instances for each provider)
# =============================================================================

_rate_limiters: Dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def get_rate_limiter(provider: str) -> RateLimiter:
    """
    Get or create a rate limiter for a specific API provider.

    Usage:
        limiter = get_rate_limiter('salesql')
        limiter.wait_if_needed()
        response = requests.get(...)
        limiter.record_request()

    Providers: crustdata, salesql, phantombuster, openai
    """
    with _limiters_lock:
        if provider not in _rate_limiters:
            config_map = {
                'crustdata': RateLimitConfig.for_crustdata,
                'salesql': RateLimitConfig.for_salesql,
                'phantombuster': RateLimitConfig.for_phantombuster,
                'openai': RateLimitConfig.for_openai,
            }
            config_fn = config_map.get(provider.lower())
            if config_fn:
                _rate_limiters[provider] = RateLimiter(config_fn())
            else:
                # Default config for unknown providers
                _rate_limiters[provider] = RateLimiter(RateLimitConfig(requests_per_minute=60))
        return _rate_limiters[provider]


def rate_limited(provider: str):
    """
    Decorator to apply rate limiting to an API call function.

    Usage:
        @rate_limited('salesql')
        def fetch_email(linkedin_url):
            return requests.get(f'https://api.salesql.com/...?url={linkedin_url}')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(provider)
            limiter.wait_if_needed()
            try:
                return func(*args, **kwargs)
            finally:
                limiter.record_request()
        return wrapper
    return decorator


# =============================================================================
# HELPER FUNCTIONS FOR COMMON API PATTERNS
# =============================================================================

def safe_api_call(
    func: Callable,
    provider: str,
    *args,
    retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
) -> tuple[Any, Optional[APIErrorResponse]]:
    """
    Execute an API call with retry logic and consistent error handling.

    Returns:
        Tuple of (result, error) where error is None on success.

    Usage:
        result, error = safe_api_call(
            requests.get,
            'crustdata',
            'https://api.crustdata.com/...',
            headers={'Authorization': 'Token xxx'},
            retries=3
        )
        if error:
            if error.is_retryable():
                # Maybe try again later
                pass
            return {'error': error.to_dict()}
        return result.json()
    """
    limiter = get_rate_limiter(provider)
    last_error = None

    for attempt in range(retries):
        try:
            limiter.wait_if_needed()
            result = func(*args, **kwargs)
            limiter.record_request()

            # Check for HTTP errors
            if hasattr(result, 'status_code'):
                if result.status_code >= 400:
                    error = APIErrorResponse.from_http_error(result, provider)
                    if error.is_retryable() and attempt < retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return None, error

            return result, None

        except RateLimitExceeded as e:
            return None, APIErrorResponse.rate_limited(provider)

        except Exception as e:
            last_error = APIErrorResponse.from_exception(e, ErrorCode.EXTERNAL_API_ERROR)
            last_error.message = f"[{provider}] {last_error.message}"

            if last_error.is_retryable() and attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue

            return None, last_error

    return None, last_error
