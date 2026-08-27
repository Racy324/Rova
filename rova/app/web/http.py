from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import httpx

from .sources import FetchedPage, SearchHit


class UnsafeUrlError(ValueError):
    pass


class WebNetworkError(RuntimeError):
    """A recoverable public-web request failure exposed to the agent."""


@dataclass(frozen=True)
class _BufferedResponse:
    status_code: int
    headers: object
    content: bytes
    body_truncated: bool = False


def validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise UnsafeUrlError("invalid URL") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("only public http/https URLs are allowed")
    if parsed.hostname.lower() == "localhost":
        raise UnsafeUrlError("localhost is not allowed")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return
    if not address.is_global:
        raise UnsafeUrlError("non-public IP addresses are not allowed")


def _resolve_public_host(hostname: str, resolver: Callable[[str], list[str]]) -> None:
    try:
        addresses = resolver(hostname)
    except OSError as error:
        raise WebNetworkError("DNS resolution failed") from error
    if not addresses:
        raise WebNetworkError("DNS resolution returned no addresses")
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as error:
            raise WebNetworkError("DNS resolution returned an invalid address") from error
        if not address.is_global:
            raise UnsafeUrlError("hostname resolves to a non-public IP address")


def _system_resolver(hostname: str) -> list[str]:
    return list({item[4][0] for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)})


class _ReadableHtmlParser(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered in self._IGNORED_TAGS:
            self._ignored_depth += 1
        if lowered == "title":
            self._in_title = True
        if lowered in {"p", "div", "br", "li", "h1", "h2", "h3", "article", "section"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        if lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        normalized = " ".join(data.split())
        if not normalized:
            return
        if self._in_title:
            self.title = f"{self.title} {normalized}".strip()
        self._parts.append(normalized)

    def extracted_text(self) -> str:
        return "\n".join(part.strip() for part in self._parts if part.strip())


class HttpWebpageFetcher:
    """Minimal, app-local HTTP GET fetcher for public textual webpages."""

    def __init__(
        self,
        client=None,
        *,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = 1_000_000,
        max_extracted_characters: int = 20_000,
        max_redirects: int = 5,
        resolver: Callable[[str], list[str]] = _system_resolver,
    ) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_extracted_characters = max_extracted_characters
        self._max_redirects = max_redirects
        self._resolver = resolver

    async def fetch(self, url: str) -> FetchedPage:
        response = await self._get_with_checked_redirects(url)
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "text/plain"}:
            label = content_type or "missing"
            raise WebNetworkError(f"unsupported content type: {label}")
        raw_content = response.content
        response_was_truncated = getattr(response, "body_truncated", False) or len(raw_content) > self._max_response_bytes
        if response_was_truncated:
            raw_content = raw_content[:self._max_response_bytes]
        encoding = getattr(response, "encoding", None) or "utf-8"
        decoded = raw_content.decode(encoding, errors="replace")
        if content_type == "text/html":
            parser = _ReadableHtmlParser()
            parser.feed(decoded)
            parser.close()
            title, content = parser.title, parser.extracted_text()
        else:
            title, content = "", decoded
        content_was_truncated = response_was_truncated or len(content) > self._max_extracted_characters
        if len(content) > self._max_extracted_characters:
            content = content[:self._max_extracted_characters]
        if content_was_truncated:
            content = f"{content}\n\ncontent_truncated=true"
        return FetchedPage(title, content)

    async def _get_with_checked_redirects(self, initial_url: str):
        url = initial_url
        for redirect_count in range(self._max_redirects + 1):
            validate_public_url(url)
            _resolve_public_host(urlsplit(url).hostname or "", self._resolver)
            response = await self._request(url)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise WebNetworkError("redirect response missing location")
                if redirect_count == self._max_redirects:
                    raise WebNetworkError("too many redirects")
                url = urljoin(url, location)
                continue
            if not 200 <= response.status_code < 300:
                raise WebNetworkError(f"HTTP {response.status_code}")
            return response
        raise WebNetworkError("too many redirects")

    async def _request(self, url: str):
        try:
            if self._client is not None:
                return await self._client.get(url, follow_redirects=False, timeout=self._timeout_seconds)
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                async with client.stream("GET", url, follow_redirects=False) as response:
                    body = bytearray()
                    body_truncated = False
                    async for chunk in response.aiter_bytes():
                        remaining = self._max_response_bytes - len(body)
                        if remaining <= 0:
                            body_truncated = True
                            break
                        body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            body_truncated = True
                            break
                    return _BufferedResponse(response.status_code, response.headers, bytes(body), body_truncated)
        except httpx.TimeoutException as error:
            raise WebNetworkError("request timeout") from error
        except httpx.RequestError as error:
            raise WebNetworkError("network request failed") from error


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._results: list[dict[str, str]] = []
        self._active: dict[str, str] | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if "result__a" in classes:
            self._active = {"title": "", "url": _unwrap_duckduckgo_url(attributes.get("href", "")), "snippet": ""}
            self._results.append(self._active)
            self._capture = "title"
        elif self._results and "result__snippet" in classes:
            self._active = self._results[-1]
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._active is None or self._capture is None:
            return
        normalized = " ".join(data.split())
        if normalized:
            field = self._capture
            self._active[field] = f"{self._active[field]} {normalized}".strip()

    def hits(self, max_results: int) -> list[SearchHit]:
        return [
            SearchHit(item["title"], item["url"], item["snippet"])
            for item in self._results
            if item["title"] and item["url"]
        ][:max_results]


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return unquote(target)
    return url


class DuckDuckGoHtmlSearchBackend:
    """Configurable, injected HTML search backend kept inside the Research app."""

    def __init__(self, endpoint: str = "https://html.duckduckgo.com/html/", *, client=None, timeout_seconds: float = 15.0) -> None:
        validate_public_url(endpoint)
        self._endpoint = endpoint
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        if max_results < 1:
            return []
        try:
            if self._client is not None:
                response = await self._client.get(
                    self._endpoint,
                    params={"q": query},
                    headers={"user-agent": "Rova-research/1.0"},
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.get(
                        self._endpoint,
                        params={"q": query},
                        headers={"user-agent": "Rova-research/1.0"},
                    )
        except httpx.TimeoutException as error:
            raise WebNetworkError("search request timeout") from error
        except httpx.RequestError as error:
            raise WebNetworkError("search network request failed") from error
        if not 200 <= response.status_code < 300:
            raise WebNetworkError(f"HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "text/html":
            raise WebNetworkError(f"unsupported search content type: {content_type or 'missing'}")
        parser = _DuckDuckGoResultParser()
        parser.feed(response.text)
        parser.close()
        return parser.hits(max_results)


DirectHttpFetchBackend = HttpWebpageFetcher
