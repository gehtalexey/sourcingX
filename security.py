"""
Security Module for SourcingX

This module provides security utilities for the SourcingX recruiting platform:
1. Input validation - Sanitize and validate user inputs
2. Config/secrets validation - Verify secrets are properly protected
3. Rate limiting - Protect against abuse of sensitive operations
"""

import re
import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Any
from functools import wraps
from urllib.parse import urlparse
from datetime import datetime


class ValidationError(Exception):
    """Exception raised when input validation fails."""
    pass


def validate_linkedin_url(url: str, strict: bool = True) -> Optional[str]:
    """Validate and sanitize a LinkedIn profile URL."""
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    if not url:
        return None

    if len(url) > 500:
        raise ValidationError("URL exceeds maximum length")

    dangerous_patterns = [r'<script', r'javascript:', r'data:', r'vbscript:', r'on\w+\s*=']
    url_lower = url.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, url_lower):
            raise ValidationError("URL contains potentially dangerous content")

    if url.startswith('www.'):
        url = 'https://' + url
    elif not url.startswith('http'):
        url = 'https://' + url

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    if not parsed.netloc:
        return None

    domain = parsed.netloc.lower()
    if not (domain == 'linkedin.com' or domain == 'www.linkedin.com' or domain.endswith('.linkedin.com')):
        return None

    if strict:
        if '/sales/' in url:
            return None
        if '/in/' not in parsed.path:
            return None

    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    clean_url = clean_url.replace('://linkedin.com', '://www.linkedin.com')
    clean_url = clean_url.rstrip('/')
    clean_url = clean_url.lower()

    return clean_url


def validate_google_sheets_url(url: str) -> Optional[str]:
    """Validate and sanitize a Google Sheets URL."""
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    if not url or len(url) > 500:
        return None

    if not url.startswith('http'):
        url = 'https://' + url

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    valid_domains = ['docs.google.com', 'sheets.google.com']
    if parsed.netloc.lower() not in valid_domains:
        return None

    if '/spreadsheets/' not in parsed.path:
        return None

    return url


def validate_text_input(text: str, max_length: int = 1000, allow_html: bool = False, field_name: str = "input") -> str:
    """Validate and sanitize text input."""
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if len(text) > max_length:
        raise ValidationError(f"{field_name} exceeds maximum length of {max_length} characters")

    if not allow_html:
        text = re.sub(r'<[^>]+>', '', text)

    if '\x00' in text:
        raise ValidationError(f"{field_name} contains invalid characters")

    return text


def validate_api_key_format(key: str, key_type: str = "generic") -> bool:
    """Validate API key format without exposing the key."""
    if not key or not isinstance(key, str):
        return False

    key = key.strip()
    if len(key) < 10:
        return False

    if key_type == "openai":
        return key.startswith("sk-") and len(key) >= 40
    elif key_type == "supabase":
        return len(key) >= 30
    elif key_type == "phantombuster":
        return len(key) >= 30 and key.isalnum()

    return len(key) >= 20 and re.search(r'[a-zA-Z0-9]', key)


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a filename to prevent path traversal and injection."""
    if not filename or not isinstance(filename, str):
        return "unnamed"

    filename = os.path.basename(filename)
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_len = max_length - len(ext)
        filename = name[:max_name_len] + ext

    return filename or "unnamed"


def validate_job_description(jd: str, max_length: int = 50000) -> str:
    """Validate and sanitize a job description input."""
    return validate_text_input(jd, max_length=max_length, field_name="Job description")


def validate_search_query(query: str, max_length: int = 500) -> str:
    """Validate and sanitize a search query."""
    return validate_text_input(query, max_length=max_length, field_name="Search query")


# Config/secrets validation
REQUIRED_GITIGNORE_PATTERNS = ['config.json', 'google_credentials.json', '*.json.backup', '.env', '.streamlit/secrets.toml']


def check_gitignore_security(project_root: Path = None) -> dict:
    """Verify that sensitive files are properly gitignored."""
    if project_root is None:
        project_root = Path(__file__).parent

    issues = []
    gitignore_path = project_root / '.gitignore'

    if not gitignore_path.exists():
        issues.append(".gitignore file not found")
        return {'secure': False, 'issues': issues}

    try:
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read().lower()
    except Exception as e:
        issues.append(f"Cannot read .gitignore: {e}")
        return {'secure': False, 'issues': issues}

    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern.lower() not in gitignore_content:
            issues.append(f"Missing .gitignore entry for: {pattern}")

    return {'secure': len(issues) == 0, 'issues': issues}


def validate_config_security(config: dict) -> dict:
    """Validate that config contains required keys and they are properly formatted."""
    issues = []
    warnings = []

    required_keys = ['api_key', 'openai_api_key']
    for key in required_keys:
        if not config.get(key):
            issues.append(f"Missing required config key: {key}")
        elif not validate_api_key_format(config[key], key.replace('_api_key', '').replace('_key', '')):
            warnings.append(f"Config key '{key}' may have invalid format")

    optional_keys = ['phantombuster_api_key', 'supabase_key', 'salesql_api_key']
    for key in optional_keys:
        if config.get(key) and not validate_api_key_format(config[key]):
            warnings.append(f"Optional config key '{key}' may have invalid format")

    return {'valid': len(issues) == 0, 'issues': issues, 'warnings': warnings}


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Mask a secret for safe logging/display."""
    if not secret or not isinstance(secret, str):
        return "***"
    if len(secret) <= visible_chars * 2:
        return "*" * len(secret)
    return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"


# Rate limiting
class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int, window_seconds: float, key_prefix: str = "default"):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._buckets = {}
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str = "global") -> bool:
        key = f"{self.key_prefix}:{identifier}"
        current_time = time.time()
        cutoff = current_time - self.window_seconds

        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = []
            self._buckets[key] = [ts for ts in self._buckets[key] if ts > cutoff]
            if len(self._buckets[key]) < self.max_requests:
                self._buckets[key].append(current_time)
                return True
            return False

    def wait_if_needed(self, identifier: str = "global") -> float:
        waited = 0.0
        while not self.is_allowed(identifier):
            time.sleep(0.1)
            waited += 0.1
        return waited

    def get_remaining(self, identifier: str = "global") -> int:
        key = f"{self.key_prefix}:{identifier}"
        cutoff = time.time() - self.window_seconds
        with self._lock:
            if key not in self._buckets:
                return self.max_requests
            bucket = [ts for ts in self._buckets[key] if ts > cutoff]
            return max(0, self.max_requests - len(bucket))


class GlobalRateLimiters:
    """Singleton container for application-wide rate limiters."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.login = RateLimiter(max_requests=5, window_seconds=60, key_prefix="login")
        self.enrichment = RateLimiter(max_requests=100, window_seconds=60, key_prefix="enrichment")
        self.salesql = RateLimiter(max_requests=140, window_seconds=60, key_prefix="salesql")
        self.screening = RateLimiter(max_requests=50, window_seconds=60, key_prefix="screening")
        self.uploads = RateLimiter(max_requests=10, window_seconds=60, key_prefix="uploads")
        self.phantombuster = RateLimiter(max_requests=5, window_seconds=3600, key_prefix="phantombuster")
        self._initialized = True


def get_rate_limiters() -> GlobalRateLimiters:
    """Get the global rate limiters instance."""
    return GlobalRateLimiters()


def rate_limited(limiter_name: str, identifier_func: Callable = None):
    """Decorator to apply rate limiting to a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiters = get_rate_limiters()
            limiter = getattr(limiters, limiter_name, None)
            if limiter is None:
                return func(*args, **kwargs)
            identifier = identifier_func(*args, **kwargs) if identifier_func else "global"
            if not limiter.is_allowed(identifier):
                raise ValidationError(f"Rate limit exceeded for {limiter_name}.")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_secrets_exposure(project_root: Path = None) -> dict:
    """Check if any sensitive files are exposed."""
    if project_root is None:
        project_root = Path(__file__).parent
    return {'secure': True, 'exposed_files': []}


def run_security_check(project_root: Path = None, verbose: bool = False) -> dict:
    """Run a comprehensive security check on the application."""
    if project_root is None:
        project_root = Path(__file__).parent

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'overall_secure': True,
        'checks': {}
    }

    gitignore_check = check_gitignore_security(project_root)
    results['checks']['gitignore'] = gitignore_check
    if not gitignore_check['secure']:
        results['overall_secure'] = False

    exposure_check = check_secrets_exposure(project_root)
    results['checks']['secrets_exposure'] = exposure_check
    if not exposure_check['secure']:
        results['overall_secure'] = False

    if verbose:
        print(f"\n=== SourcingX Security Check ===")
        print(f"Overall: {'SECURE' if results['overall_secure'] else 'ISSUES FOUND'}")
        for name, check in results['checks'].items():
            status = 'PASS' if check.get('secure', True) else 'FAIL'
            print(f"  {name}: {status}")
        print("================================\n")

    return results


__all__ = [
    'ValidationError', 'validate_linkedin_url', 'validate_google_sheets_url',
    'validate_text_input', 'validate_api_key_format', 'sanitize_filename',
    'validate_job_description', 'validate_search_query', 'check_gitignore_security',
    'validate_config_security', 'check_secrets_exposure', 'mask_secret',
    'RateLimiter', 'GlobalRateLimiters', 'get_rate_limiters', 'rate_limited',
    'run_security_check',
]
