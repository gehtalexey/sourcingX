"""
Error Handling Patterns for SourcingX

This module provides:
1. Custom exception hierarchy for typed error handling
2. Retry decorator with exponential backoff for transient failures
3. Circuit breaker pattern to prevent cascade failures

Usage:
    from error_handling import (
        ApplicationError, ExternalServiceError, RateLimitError,
        retry_with_backoff, CircuitBreaker
    )
"""

import time
import functools
import threading
from typing import Callable, TypeVar, Any, Optional, Tuple, Type
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# =============================================================================
# 1. CUSTOM EXCEPTION HIERARCHY
# =============================================================================

class ApplicationError(Exception):
    """Base exception for all application errors.

    Provides structured error information including error codes,
    user-friendly messages, and context for debugging.
    """

    def __init__(self, message: str, code: str = None, details: dict = None,
                 recoverable: bool = True, original_error: Exception = None):
        """
        Args:
            message: Human-readable error message
            code: Error code for programmatic handling (e.g., "RATE_LIMIT_EXCEEDED")
            details: Additional context for debugging
            recoverable: Whether the operation can be retried
            original_error: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.code = code or "APPLICATION_ERROR"
        self.details = details or {}
        self.recoverable = recoverable
        self.original_error = original_error
        self.timestamp = datetime.utcnow()

    def __str__(self):
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ExternalServiceError(ApplicationError):
    """Error when communicating with external services (APIs)."""

    def __init__(self, service: str, message: str, status_code: int = None,
                 response_body: str = None, **kwargs):
        """
        Args:
            service: Name of the external service (e.g., "OpenAI", "Crustdata")
            message: Error description
            status_code: HTTP status code if applicable
            response_body: Raw response body for debugging
        """
        self.service = service
        self.status_code = status_code
        self.response_body = response_body

        code = kwargs.pop("code", f"{service.upper()}_ERROR")
        details = kwargs.pop("details", {})
        details.update({
            "service": service,
            "status_code": status_code,
        })

        super().__init__(message, code=code, details=details, **kwargs)


class RateLimitError(ExternalServiceError):
    """Error when API rate limits are exceeded."""

    def __init__(self, service: str, retry_after: float = None, **kwargs):
        """
        Args:
            service: Name of the rate-limited service
            retry_after: Seconds to wait before retrying (if known)
        """
        self.retry_after = retry_after
        message = kwargs.pop("message", f"{service} rate limit exceeded")
        if retry_after:
            message += f" (retry after {retry_after}s)"

        super().__init__(
            service=service,
            message=message,
            status_code=429,
            code=f"{service.upper()}_RATE_LIMIT",
            recoverable=True,
            **kwargs
        )


class QuotaExceededError(ExternalServiceError):
    """Error when API quotas/credits are exhausted."""

    def __init__(self, service: str, quota_type: str = "credits", **kwargs):
        """
        Args:
            service: Name of the service
            quota_type: Type of quota (e.g., "credits", "daily_limit")
        """
        self.quota_type = quota_type
        message = kwargs.pop("message", f"{service} {quota_type} exhausted")

        super().__init__(
            service=service,
            message=message,
            code=f"{service.upper()}_QUOTA_EXCEEDED",
            recoverable=False,  # Quotas don't reset with retries
            **kwargs
        )


class AuthenticationError(ExternalServiceError):
    """Error when API authentication fails."""

    def __init__(self, service: str, **kwargs):
        message = kwargs.pop("message", f"{service} authentication failed - check API key")

        super().__init__(
            service=service,
            message=message,
            status_code=kwargs.get("status_code", 401),
            code=f"{service.upper()}_AUTH_ERROR",
            recoverable=False,
            **kwargs
        )


class ServiceUnavailableError(ExternalServiceError):
    """Error when external service is down or unreachable."""

    def __init__(self, service: str, **kwargs):
        message = kwargs.pop("message", f"{service} is temporarily unavailable")

        super().__init__(
            service=service,
            message=message,
            status_code=kwargs.get("status_code", 503),
            code=f"{service.upper()}_UNAVAILABLE",
            recoverable=True,
            **kwargs
        )


class ValidationError(ApplicationError):
    """Error for invalid input data."""

    def __init__(self, field: str, message: str, value: Any = None, **kwargs):
        self.field = field
        self.invalid_value = value

        details = kwargs.pop("details", {})
        details.update({"field": field})
        if value is not None:
            details["invalid_value"] = str(value)[:100]

        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            recoverable=False,
            **kwargs
        )


class DataProcessingError(ApplicationError):
    """Error during data processing/transformation."""

    def __init__(self, operation: str, message: str, **kwargs):
        self.operation = operation

        details = kwargs.pop("details", {})
        details["operation"] = operation

        super().__init__(
            message=message,
            code="DATA_PROCESSING_ERROR",
            details=details,
            **kwargs
        )


# =============================================================================
# 2. RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        RateLimitError,
        ServiceUnavailableError,
        ConnectionError,
        TimeoutError,
    ),
    on_retry: Callable[[Exception, int, float], None] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not including initial try)
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries (caps exponential growth)
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exception types that trigger retries
        on_retry: Callback function(exception, attempt, delay) called before each retry

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_api():
            response = requests.get(url)
            if response.status_code == 429:
                raise RateLimitError("API")
            return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Don't retry if we've exhausted retries
                    if attempt >= max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Check if exception has retry_after hint
                    if hasattr(e, 'retry_after') and e.retry_after:
                        delay = max(delay, e.retry_after)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random())

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    raise

            # Exhausted all retries
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def create_retry_decorator_for_service(
    service_name: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    logger: Callable[[str], None] = None,
) -> Callable:
    """
    Factory function to create service-specific retry decorators.

    Args:
        service_name: Name of the service (for logging)
        max_retries: Maximum retry attempts
        base_delay: Initial delay
        logger: Optional logging function

    Returns:
        Configured retry decorator
    """
    def log_retry(exc: Exception, attempt: int, delay: float):
        msg = f"[{service_name}] Retry {attempt}/{max_retries} after {delay:.1f}s: {exc}"
        if logger:
            logger(msg)
        else:
            print(msg)

    return retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        on_retry=log_retry,
    )


# =============================================================================
# 3. CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"      # Normal operation - requests allowed
    OPEN = "open"          # Failure threshold exceeded - requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5       # Failures before opening circuit
    success_threshold: int = 2       # Successes in half-open before closing
    timeout: float = 60.0            # Seconds before transitioning open -> half-open
    half_open_max_calls: int = 3     # Max calls allowed in half-open state


class CircuitOpenError(ApplicationError):
    """Raised when circuit breaker is open and blocking requests."""

    def __init__(self, service: str, time_until_retry: float = None):
        message = f"Circuit breaker open for {service}"
        if time_until_retry:
            message += f" - retry in {time_until_retry:.0f}s"

        super().__init__(
            message=message,
            code="CIRCUIT_OPEN",
            details={"service": service, "time_until_retry": time_until_retry},
            recoverable=True,
        )
        self.service = service
        self.time_until_retry = time_until_retry


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.

    Prevents cascade failures by stopping requests to failing services
    and allowing them to recover.

    Usage:
        # Create a circuit breaker for OpenAI
        openai_circuit = CircuitBreaker("OpenAI")

        # Wrap API calls
        with openai_circuit:
            response = openai_client.chat.completions.create(...)

        # Or use as decorator
        @openai_circuit
        def call_openai():
            return openai_client.chat.completions.create(...)

    States:
        CLOSED: Normal operation, all requests pass through
        OPEN: Service failing, requests are blocked
        HALF_OPEN: Testing recovery, limited requests allowed
    """

    # Class-level registry of all circuit breakers (for monitoring)
    _registry: dict = {}
    _registry_lock = threading.Lock()

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] = None,
    ):
        """
        Args:
            name: Identifier for this circuit breaker (e.g., "OpenAI", "Crustdata")
            config: Configuration options
            on_state_change: Callback when state changes (name, old_state, new_state)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # State tracking (thread-safe)
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

        # Register this circuit breaker
        with CircuitBreaker._registry_lock:
            CircuitBreaker._registry[name] = self

    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state."""
        with self._lock:
            return self._state

    @classmethod
    def get_all_states(cls) -> dict:
        """Get states of all registered circuit breakers."""
        with cls._registry_lock:
            return {name: cb.get_status() for name, cb in cls._registry.items()}

    @classmethod
    def get_circuit(cls, name: str) -> Optional["CircuitBreaker"]:
        """Get a circuit breaker by name."""
        with cls._registry_lock:
            return cls._registry.get(name)

    def get_status(self) -> dict:
        """Get detailed status of this circuit breaker."""
        with self._lock:
            status = {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
            }

            if self._state == CircuitState.OPEN and self._last_failure_time:
                elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                time_until_retry = max(0, self.config.timeout - elapsed)
                status["time_until_retry"] = time_until_retry

            return status

    def _change_state(self, new_state: CircuitState):
        """Change state and trigger callback."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            if self.on_state_change:
                self.on_state_change(self.name, old_state, new_state)

    def _check_state(self):
        """Check if state should transition (called under lock)."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._last_failure_time:
                elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                if elapsed >= self.config.timeout:
                    self._change_state(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    self._success_count = 0

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            self._check_state()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN state - check if we can transition
            return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    # Service recovered, close circuit
                    self._change_state(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exception: Exception = None):
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                # Failure during recovery test, reopen circuit
                self._change_state(CircuitState.OPEN)
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    self._change_state(CircuitState.OPEN)

    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._change_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def __enter__(self):
        """Context manager entry - check if request is allowed."""
        if not self.allow_request():
            status = self.get_status()
            raise CircuitOpenError(
                self.name,
                time_until_retry=status.get("time_until_retry")
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success or failure."""
        if exc_type is None:
            self.record_success()
        else:
            # Only record failure for certain exception types
            if isinstance(exc_val, (
                ExternalServiceError,
                ConnectionError,
                TimeoutError,
            )):
                self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use circuit breaker as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self:
                return func(*args, **kwargs)
        return wrapper


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_http_error(status_code: int, service: str, response_body: str = None) -> ExternalServiceError:
    """
    Convert HTTP status codes to appropriate exception types.

    Args:
        status_code: HTTP response status code
        service: Name of the service
        response_body: Optional response body for context

    Returns:
        Appropriate ExternalServiceError subclass
    """
    if status_code == 429:
        return RateLimitError(service, response_body=response_body)
    elif status_code == 401 or status_code == 403:
        return AuthenticationError(service, status_code=status_code, response_body=response_body)
    elif status_code == 402:
        return QuotaExceededError(service, response_body=response_body)
    elif status_code >= 500:
        return ServiceUnavailableError(service, status_code=status_code, response_body=response_body)
    else:
        return ExternalServiceError(
            service=service,
            message=f"HTTP {status_code} error",
            status_code=status_code,
            response_body=response_body,
        )


# Pre-configured circuit breakers for common services
# These are created lazily on first use
_service_circuits: dict = {}
_service_circuits_lock = threading.Lock()


def get_service_circuit(
    service_name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service.

    Args:
        service_name: Name of the service
        failure_threshold: Failures before opening
        timeout: Seconds before attempting recovery

    Returns:
        CircuitBreaker instance for the service
    """
    with _service_circuits_lock:
        if service_name not in _service_circuits:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                timeout=timeout,
            )
            _service_circuits[service_name] = CircuitBreaker(service_name, config)
        return _service_circuits[service_name]


# Export all public classes and functions
__all__ = [
    # Exceptions
    "ApplicationError",
    "ExternalServiceError",
    "RateLimitError",
    "QuotaExceededError",
    "AuthenticationError",
    "ServiceUnavailableError",
    "ValidationError",
    "DataProcessingError",
    "CircuitOpenError",
    # Retry
    "retry_with_backoff",
    "create_retry_decorator_for_service",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    # Utilities
    "classify_http_error",
    "get_service_circuit",
]
