"""Small stdlib HTTP client with timeouts and one retry."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import __version__

USER_AGENT = f"KillerFish-collector/{__version__} (campus transit research)"


@dataclass
class Response:
    url: str
    status: int  # 0 = network error
    body: bytes
    elapsed_ms: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and not self.error

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def fetch(url: str, timeout: float = 15.0, retries: int = 1, data: bytes | None = None,
          headers: dict[str, str] | None = None) -> Response:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    last = Response(url, 0, b"", 0, "not attempted")
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return Response(url, r.status, body, int((time.monotonic() - start) * 1000))
        except urllib.error.HTTPError as e:
            body = e.read() if hasattr(e, "read") else b""
            last = Response(url, e.code, body, int((time.monotonic() - start) * 1000), f"HTTP {e.code}")
            if e.code < 500 and e.code != 429:
                return last
        except Exception as e:  # network errors, timeouts
            last = Response(url, 0, b"", int((time.monotonic() - start) * 1000), f"{type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return last


class Oba:
    """OneBusAway REST client: 'wayfinder' keyless proxy or 'standard' /api/where with a key."""

    def __init__(self, base_url: str, flavor: str = "wayfinder", api_key: str = ""):
        self.base = base_url if base_url.endswith("/") else base_url + "/"
        self.flavor = flavor
        self.api_key = api_key

    def url(self, method: str, ident: str | None = None, **params) -> str:
        path = method + ("/" + urllib.parse.quote(ident, safe="") if ident else "")
        if self.flavor == "standard":
            path += ".json"
            if self.api_key:
                params["key"] = self.api_key
        q = urllib.parse.urlencode({k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()})
        return self.base + path + ("?" + q if q else "")

    def get(self, method: str, ident: str | None = None, **params) -> Response:
        return fetch(self.url(method, ident, **params))
