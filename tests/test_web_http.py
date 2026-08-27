from __future__ import annotations

import pytest
import httpx

from rova.app.web.sources import SearchHit
from rova.app.web.http import DuckDuckGoHtmlSearchBackend, HttpWebpageFetcher, UnsafeUrlError, WebNetworkError, validate_public_url


@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://[::1]/", "http://169.254.1.1/",
    "http://10.0.0.1/", "http://192.168.1.1/", "file:///tmp/test",
])
def test_public_url_validation_rejects_local_or_non_http_targets(url):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


def test_public_url_validation_accepts_public_http_url():
    validate_public_url("https://docs.python.org/3/")


@pytest.mark.asyncio
async def test_fetcher_extracts_html_text_and_marks_truncation():
    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        url = "https://example.test/"
        content = b"<html><title>Example</title><body><p>first readable paragraph</p><script>ignored</script><p>second</p></body></html>"

    class Client:
        async def get(self, url, follow_redirects=False, timeout=None):
            return Response()

    page = await HttpWebpageFetcher(Client(), max_extracted_characters=20, resolver=lambda _: ["8.8.8.8"]).fetch("https://example.test/")
    assert page.title == "Example"
    assert "content_truncated=true" in page.content


@pytest.mark.asyncio
async def test_fetcher_maps_http_error_to_web_network_error():
    class Response:
        status_code = 404
        headers = {"content-type": "text/html"}
        url = "https://example.test/missing"
        content = b"missing"

    class Client:
        async def get(self, url, follow_redirects=False, timeout=None):
            return Response()

    with pytest.raises(WebNetworkError, match="HTTP 404"):
        await HttpWebpageFetcher(Client(), resolver=lambda _: ["8.8.8.8"]).fetch("https://example.test/missing")


@pytest.mark.asyncio
async def test_fetcher_maps_timeout_to_safe_network_error():
    class Client:
        async def get(self, url, follow_redirects=False, timeout=None):
            raise httpx.ReadTimeout("socket stalled", request=httpx.Request("GET", url))

    with pytest.raises(WebNetworkError, match="request timeout"):
        await HttpWebpageFetcher(Client(), resolver=lambda _: ["8.8.8.8"]).fetch("https://example.test/")


@pytest.mark.asyncio
async def test_fetcher_rejects_unsupported_content_type():
    class Response:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = b"%PDF"

    class Client:
        async def get(self, url, follow_redirects=False, timeout=None):
            return Response()

    with pytest.raises(WebNetworkError, match="unsupported content type: application/pdf"):
        await HttpWebpageFetcher(Client(), resolver=lambda _: ["8.8.8.8"]).fetch("https://example.test/file")


@pytest.mark.asyncio
async def test_fetcher_revalidates_each_redirect_target():
    class Response:
        status_code = 302
        headers = {"location": "http://127.0.0.1/private"}
        content = b""

    class Client:
        async def get(self, url, follow_redirects=False, timeout=None):
            return Response()

    with pytest.raises(UnsafeUrlError):
        await HttpWebpageFetcher(Client(), resolver=lambda _: ["8.8.8.8"]).fetch("https://example.test/start")


@pytest.mark.asyncio
async def test_search_backend_parses_public_html_results_without_exposing_configuration():
    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = """<div class=\"result\"><a class=\"result__a\" href=\"https://packaging.python.org/guides/\">Packaging guide</a><a class=\"result__snippet\">pyproject configuration</a></div>"""

    class Client:
        async def get(self, url, **kwargs):
            assert url == "https://search.example.test/html/"
            assert kwargs["params"] == {"q": "pyproject.toml"}
            return Response()

    hits = await DuckDuckGoHtmlSearchBackend("https://search.example.test/html/", client=Client()).search("pyproject.toml", 3)

    assert hits == [SearchHit("Packaging guide", "https://packaging.python.org/guides/", "pyproject configuration")]


@pytest.mark.asyncio
async def test_search_backend_maps_http_failure_to_safe_network_error():
    class Response:
        status_code = 429
        headers = {"content-type": "text/html"}

    class Client:
        async def get(self, url, **kwargs):
            return Response()

    with pytest.raises(WebNetworkError, match="HTTP 429"):
        await DuckDuckGoHtmlSearchBackend("https://search.example.test/html/", client=Client()).search("pyproject", 3)
