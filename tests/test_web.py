import pytest
from requests import Response

from aletheia.config import WebConfig
from aletheia.web import WebAccessError, WebExplorer


def test_parser_and_link_discovery():
    html = '<html><head><title>T</title></head><body><p>Hello <b>world</b></p><a href="/x">x</a><script>bad</script></body></html>'
    assert "hello world" in WebExplorer.extract_text(html).lower()
    assert WebExplorer.extract_links(html, "https://example.org/a") == ["https://example.org/x"]


def test_xml_feed_links_and_title():
    xml = '<?xml version="1.0"?><rss><channel><title>Feed</title><item><title>Article</title><link>https://example.org/article</link></item></channel></rss>'
    assert WebExplorer.page_title(xml, xml=True) == "Feed"
    assert WebExplorer.extract_feed_links(xml, "https://example.org/feed") == ["https://example.org/article"]
    assert "Feed Article" in WebExplorer.extract_text(xml, xml=True)


def test_rejects_local_urls():
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 0)
    with pytest.raises(ValueError):
        web.open("file:///tmp/x")
    assert web.normalize_url("http://127.0.0.1:8000/") is None
    web.close()


def test_search_records_provider_errors(monkeypatch):
    from aletheia.config import WebSearchProvider
    config = WebConfig([WebSearchProvider(name="test", url="https://example.org/search", result_selector="a")], [], ["http", "https"])
    web = WebExplorer(config, 1, 1000, "test", 0, 0)
    monkeypatch.setattr(web, "_search_with_provider", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("search unavailable")))
    assert web.search("hello") == []
    assert web.last_errors and "search unavailable" in web.last_errors[0]
    web.close()


def test_retry_after_backoff(monkeypatch):
    web = WebExplorer(WebConfig([], [], ["http", "https"]), 1, 1000, "test", 0, 1)
    monkeypatch.setattr(web._dns_adapter, "_resolve_public", lambda host: "93.184.216.34")
    monkeypatch.setattr(web, "_host_is_public", lambda host: True)
    calls = []

    def fake_get(*args, **kwargs):
        response = Response()
        response.status_code = 429
        response.url = "https://example.org"
        response.headers["Retry-After"] = "1"
        calls.append(1)
        return response

    monkeypatch.setattr(web.session, "get", fake_get)
    monkeypatch.setattr("aletheia.web.time.sleep", lambda _: None)
    response = web._get("https://example.org")
    assert response.status_code == 429
    assert len(calls) == 2
    web.close()


def test_memory_blocks_access_denied(tmp_path):
    from aletheia.memory import ExperienceMemory
    m = ExperienceMemory(tmp_path / "m.sqlite3", 10, 10)
    url = "https://example.org/blocked"
    m.add_links([url], "https://example.org", 1)
    assert m.next_link()["url"] == url
    m.block_url(url, "robots.txt disallows this URL", source="test")
    assert m.next_link() is None
    assert m.is_blocked(url)
    assert m.summary()["blocked_resources"] == 1
    m.close()


def test_access_error_is_specific():
    error = WebAccessError("https://example.org/x", "robots policy")
    assert error.url.endswith("/x")
    assert "robots policy" in str(error)



def test_pinned_http_connection_uses_urllib3_connector():
    import socket
    import threading
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from requests import Session
    from aletheia.web import _PinnedDNSAdapter

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b"pinned-http-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = _PinnedDNSAdapter()
        adapter._resolve_public = lambda host: "127.0.0.1"
        session = Session()
        session.trust_env = False
        session.mount("http://", adapter)
        response = session.get(f"http://example.test:{server.server_port}/", timeout=3)
        assert response.status_code == 200
        assert response.text == "pinned-http-ok"
        session.close()
    finally:
        server.shutdown()
        server.server_close()


def test_pinned_https_pool_preserves_hostname_and_tls_settings():
    from urllib3 import PoolManager
    from aletheia.web import _PinnedHTTPSPool
    import ssl

    context = ssl.create_default_context()
    pool = _PinnedHTTPSPool(
        "example.test",
        443,
        pinned_ip="93.184.216.34",
        ssl_context=context,
        assert_hostname="example.test",
        server_hostname="example.test",
    )
    assert pool.conn_kw["pinned_ip"] == "93.184.216.34"
    assert pool.assert_hostname == "example.test"
    assert pool.conn_kw["server_hostname"] == "example.test"
    pool.close()


def test_pinned_adapter_rejects_loopback_resolution():
    from aletheia.web import _PinnedDNSAdapter
    adapter = _PinnedDNSAdapter()
    with pytest.raises(ValueError):
        adapter._resolve_public("localhost")


def test_pinned_https_request_uses_pinned_ip_and_original_hostname(tmp_path):
    import shutil
    import ssl
    import subprocess
    import tempfile
    import threading
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from pathlib import Path
    from requests import Session
    from aletheia.web import _PinnedDNSAdapter

    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl is required for the local HTTPS integration test")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = b"pinned-https-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *_):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = root / "key.pem"
        cert = root / "cert.pem"
        conf = root / "openssl.cnf"
        conf.write_text(
            "[req]\n"
            "distinguished_name=req_distinguished_name\n"
            "req_extensions=req_ext\n"
            "prompt=no\n"
            "[req_distinguished_name]\n"
            "CN=example.test\n"
            "[req_ext]\n"
            "subjectAltName=DNS:example.test\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                openssl, "req", "-x509", "-nodes", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(cert), "-days", "1",
                "-config", str(conf), "-extensions", "req_ext",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert, keyfile=key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = _PinnedDNSAdapter()
            adapter._resolve_public = lambda host: "127.0.0.1"
            session = Session()
            session.trust_env = False
            session.mount("https://", adapter)
            response = session.get(
                f"https://example.test:{server.server_port}/",
                timeout=3,
                verify=str(cert),
            )
            assert response.status_code == 200
            assert response.text == "pinned-https-ok"
            session.close()
        finally:
            server.shutdown()
            server.server_close()


def test_pinned_adapter_rejects_shared_cgnat_dns_answer(monkeypatch):
    import socket
    from aletheia.web import _PinnedDNSAdapter

    def fake_getaddrinfo(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.64.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="non-public or special-use"):
        _PinnedDNSAdapter._resolve_public("example.test")
