# lic/retry_utils.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type


# A conservative set of exception name substrings we often see in OpenAI/client stacks.
_TRANSIENT_SUBSTRINGS = [
    "Timeout",
    "timed out",
    "ReadTimeout",
    "ConnectTimeout",
    "APIConnectionError",
    "RateLimit",
    "rate limit",
    "ServiceUnavailable",
    "InternalServerError",
    "Bad gateway",
    "502",
    "503",
    "504",
]


def _looks_transient(exc: BaseException) -> bool:
    s = repr(exc)
    return any(tok in s for tok in _TRANSIENT_SUBSTRINGS)


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay_s: float = 0.8
    max_delay_s: float = 10.0
    jitter: float = 0.25  # +/- fraction


def retry_call(
    fn: Callable[[], Any],
    *,
    cfg: RetryConfig,
    retry_on: Optional[Tuple[Type[BaseException], ...]] = None,
    retry_if: Optional[Callable[[BaseException], bool]] = None,
) -> Any:
    """
    Retry a zero-arg callable with exponential backoff + jitter.

    - retry_on: explicit exception types to retry
    - retry_if: predicate deciding retry based on exception (used in addition to retry_on)
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as e:
            do_retry = False
            if retry_on is not None and isinstance(e, retry_on):
                do_retry = True
            if retry_if is not None and retry_if(e):
                do_retry = True
            # default heuristic if neither provided
            if retry_on is None and retry_if is None and _looks_transient(e):
                do_retry = True

            if (not do_retry) or attempt >= cfg.max_attempts:
                raise

            # exponential backoff
            delay = min(cfg.base_delay_s * (2 ** (attempt - 1)), cfg.max_delay_s)
            # jitter
            jitter = delay * cfg.jitter * (2 * random.random() - 1.0)
            time.sleep(max(0.0, delay + jitter))
