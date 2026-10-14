from __future__ import annotations

import ipaddress
import random
import re
import socket
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from pathlib import Path
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy, prepend_scheme_if_needed
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.util.connection import create_connection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from bs4 import BeautifulSoup

from .config import DiscoverySource, WebConfig, WebSearchProvider
from .documents import read_document_bytes




class _PinnedHTTPConnection(HTTPConnection):
    def __init__(self, *args, pinned_ip: str, **kwargs):
        self._pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self):
        return create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            source_address=self.source_address,
            socket_options=self.socket_options,
        )


class _PinnedHTTPSConnection(HTTPSConnection):
    def __init__(self, *args, pinned_ip: str, **kwargs):
        self._pinned_ip = pinned_ip
        super().__init__(*args, **kwargs)

    def _new_conn(self):
        return create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            source_address=self.source_address,
            socket_options=self.socket_options,
        )


class _PinnedHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PinnedHTTPConnection

    def __init__(self, host: str, port: int, *, pinned_ip: str, **kwargs):
        kwargs["pinned_ip"] = pinned_ip
        super().__init__(host, port, **kwargs)


class _PinnedHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection

    def __init__(self, host: str, port: int, *, pinned_ip: str, **kwargs):
        kwargs["pinned_ip"] = pinned_ip
        kwargs.setdefault("server_hostname", host)
        kwargs.setdefault("assert_hostname", host)
        super().__init__(host, port, **kwargs)


class _PinnedDNSAdapter(HTTPAdapter):
    """Connect to the exact public IP resolved immediately before the request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pins: dict[str, str] = {}

    @staticmethod
    def _resolve_public(host: str) -> str:
        addresses = []
        for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
            address = item[4][0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            # ``is_global`` also excludes shared CGNAT, documentation ranges, and
            # other special-use networks that are not consistently covered by the
            # individual private/loopback/reserved flags.
            if not ip.is_global:
                raise ValueError("DNS resolved to a non-public or special-use address")
            addresses.append(address)
        if not addresses:
            raise ValueError("host has no public address")
        return sorted(set(addresses))[0]

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        if select_proxy(request.url, proxies):
            raise ValueError("HTTP proxies are disabled for pinned outbound requests")
        host_params, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
        host = host_params["host"]
        port = int(host_params.get("port") or (443 if host_params["scheme"] == "https" else 80))
        pinned_ip = self._pins.pop(host.lower().rstrip("."), None)
        if pinned_ip is None:
            pinned_ip = self._resolve_public(host)
        pool_kwargs = dict(pool_kwargs)
        if host_params["scheme"] == "https":
            return _PinnedHTTPSPool(host, port, pinned_ip=pinned_ip, **pool_kwargs)
        # Requests may pass TLS-only connection settings even for an HTTP
        # request. urllib3's HTTPConnection rejects those keywords.
        for key in (
            "cert_reqs", "ssl_context", "ca_certs", "ca_cert_dir",
            "cert_file", "key_file", "key_password", "assert_hostname",
            "assert_fingerprint", "server_hostname", "ssl_minimum_version",
            "ssl_maximum_version", "ssl_version",
        ):
            pool_kwargs.pop(key, None)
        return _PinnedHTTPPool(host, port, pinned_ip=pinned_ip, **pool_kwargs)

    def pin_host(self, host: str, ip: str) -> None:
        self._pins[host.lower().rstrip(".")] = ip

class WebAccessError(PermissionError):
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"{reason}: {url}")


@dataclass
class Page:
    url: str
    title: str
    text: str
    status_code: int
    links: list[str]
    content_type: str = "text/html"
    depth: int = 0
    quality_report: dict = field(default_factory=dict)


class WebExplorer:
    def __init__(self, config: WebConfig, timeout: float, max_chars: int, user_agent: str, min_host_interval: float, max_retries: int, max_redirects: int = 5, max_download_bytes: int = 50 * 1024 * 1024, max_pdf_chars: int = 5_000_000, default_text_encoding: str = "utf-8", crawl_exclude_patterns: list[str] | None = None, dns_cache_seconds: float = 30.0):
        self.config = config
        self.timeout = float(timeout)
        self.max_chars = int(max_chars)
        self.min_host_interval = max(0.0, float(min_host_interval))
        self.max_retries = max(0, int(max_retries))
        self.max_redirects = max(0, int(max_redirects))
        self.max_download_bytes = max(1, int(max_download_bytes))
        self.max_pdf_chars = max(1, int(max_pdf_chars))
        self.default_text_encoding = str(default_text_encoding or "utf-8")
        self.crawl_exclude_patterns = [re.compile(str(pattern), re.IGNORECASE) for pattern in (crawl_exclude_patterns or []) if str(pattern).strip()]
        self.dns_cache_seconds = max(0.0, float(dns_cache_seconds))
        self.session = requests.Session()
        self.session.trust_env = False
        adapter = _PinnedDNSAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        self._dns_adapter = adapter
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.8"})
        self._last_request: dict[str, float] = {}
        self._provider_index = 0
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._public_host_cache: dict[str, tuple[float, bool, str | None]] = {}
        self.last_errors: list[str] = []

    @staticmethod
    def normalize_url(url: str, base_url: str | None = None, allowed_schemes: list[str] | None = None) -> str | None:
        if not url:
            return None
        if base_url:
            url = urllib.parse.urljoin(base_url, url)
        parsed = urllib.parse.urlparse(url)
        allowed = set(allowed_schemes or ["http", "https"])
        if parsed.scheme not in allowed or not parsed.netloc:
            return None
        host = parsed.hostname
        if not host:
            return None
        # Syntactic checks are applied here; DNS-backed private-host checks are
        # enforced by the instance before network access.
        normalized_host = host.strip("[]").lower().rstrip(".")
        if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
            return None
        try:
            ip = ipaddress.ip_address(normalized_host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                return None
        except ValueError:
            pass
        clean = parsed._replace(fragment="").geturl()
        return clean[:4000]

    def _host_is_public(self, host: str) -> bool:
        normalized = host.strip("[]").lower().rstrip(".")
        if normalized in {"localhost", "localhost.localdomain", "broadcasthost"} or normalized.endswith(".localhost"):
            return False
        now = time.monotonic()
        cached = self._public_host_cache.get(normalized)
        cache_seconds = self.dns_cache_seconds
        if cached and cached[0] > now:
            return bool(cached[1])
        try:
            ip = ipaddress.ip_address(normalized)
            allowed = bool(ip.is_global)
            self._public_host_cache[normalized] = (now + cache_seconds, allowed, normalized if allowed else None)
            return allowed
        except ValueError:
            pass
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)}
        except (socket.gaierror, OSError):
            self._public_host_cache[normalized] = (now + min(cache_seconds, 5.0), False, None)
            return False
        if not addresses:
            self._public_host_cache[normalized] = (now + min(cache_seconds, 5.0), False, None)
            return False
        # Every returned address must be globally routable. This also rejects
        # CGNAT, documentation, link-local, loopback, multicast and RFC 1918 space.
        allowed = True
        representative = None
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                allowed = False
                break
            if not ip.is_global:
                allowed = False
                break
            representative = representative or address
        self._public_host_cache[normalized] = (now + cache_seconds, allowed, representative)
        return allowed

    def _host(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    def public_addresses(self, host: str) -> list[str]:
        normalized = str(host or "").strip("[]").lower().rstrip(".")
        if not self._host_is_public(normalized):
            return []
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)}
        except (socket.gaierror, OSError):
            return []
        return sorted(addresses)

    def _pace(self, host: str) -> None:
        wait = self.min_host_interval - (time.monotonic() - self._last_request.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_delay(response: requests.Response | None, delay: float) -> float:
        if response is not None:
            value = response.headers.get("Retry-After")
            if value:
                try:
                    return min(90.0, max(1.0, float(value)))
                except ValueError:
                    try:
                        return min(90.0, max(1.0, parsedate_to_datetime(value).timestamp() - time.time()))
                    except Exception:
                        pass
        return min(90.0, delay)

    def _get(self, url: str, *, params: dict | None = None) -> requests.Response:
        current = self.normalize_url(url, allowed_schemes=self.config.allowed_schemes)
        if not current:
            raise ValueError("URL must be absolute and use an allowed public HTTP(S) scheme")
        delay = 2.0
        for redirect_index in range(self.max_redirects + 1):
            parsed = urllib.parse.urlparse(current)
            host_name = parsed.hostname or ""
            host = self._host(current)
            for attempt in range(self.max_retries + 1):
                self._pace(host)
                try:
                    pinned_ip = self._dns_adapter._resolve_public(host_name)
                    self._dns_adapter.pin_host(host_name, pinned_ip)
                    try:
                        response = self.session.get(
                            current, params=params if redirect_index == 0 else None,
                            timeout=self.timeout, allow_redirects=False, stream=True,
                        )
                    finally:
                        self._dns_adapter._pins.pop(host_name.lower().rstrip("."), None)
                    self._last_request[host] = time.monotonic()
                    if response.is_redirect or response.status_code in {301,302,303,307,308}:
                        location = response.headers.get("Location")
                        if getattr(response, "raw", None) is not None:
                            response.close()
                        if not location:
                            raise ValueError("redirect response has no Location header")
                        next_url = self.normalize_url(location, current, self.config.allowed_schemes)
                        if not next_url:
                            raise ValueError("redirect target uses a disallowed scheme or URL")
                        next_host = urllib.parse.urlparse(next_url).hostname or ""
                        if not self._host_is_public(next_host):
                            raise ValueError("redirect target resolves to a non-public or unresolvable host")
                        if not self._robots(next_url):
                            raise WebAccessError(next_url, "robots.txt disallows redirected URL")
                        current = next_url
                        break
                    if response.status_code not in {408,425,429,500,502,503,504}:
                        length = response.headers.get("Content-Length")
                        if length:
                            try:
                                declared_length = int(length)
                            except (TypeError, ValueError):
                                declared_length = None
                            if declared_length is not None and declared_length > self.max_download_bytes:
                                response.close()
                                raise ValueError(
                                    f"response exceeds download limit of {self.max_download_bytes} bytes"
                                )
                        chunks: list[bytes] = []
                        total = 0
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            total += len(chunk)
                            if total > self.max_download_bytes:
                                response.close()
                                raise ValueError(
                                    f"response exceeds download limit of {self.max_download_bytes} bytes"
                                )
                            chunks.append(chunk)
                        response._content = b"".join(chunks)
                        response._content_consumed = True
                        response.url = current
                        return response
                    if attempt >= self.max_retries:
                        response.url = current
                        return response
                    time.sleep(self._retry_delay(response, delay))
                    if getattr(response, "raw", None) is not None:
                        response.close()
                    delay *= 2.0
                except requests.RequestException:
                    self._dns_adapter._pins.pop(host_name.lower().rstrip("."), None)
                    self._last_request[host] = time.monotonic()
                    if attempt >= self.max_retries:
                        raise
                    time.sleep(min(90.0, delay))
                    delay *= 2.0
            else:
                raise RuntimeError("web request failed")
            if redirect_index >= self.max_redirects:
                raise ValueError("too many HTTP redirects")
            continue
        raise RuntimeError("web request failed")

    def _robots(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._robots_cache.get(origin)
        if parser is None:
            robots_url = origin + "/robots.txt"
            try:
                response = self._get(robots_url)
                if response.status_code >= 400:
                    return True
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())
                self._robots_cache[origin] = parser
            except Exception:
                return True
        try:
            return parser.can_fetch(self.session.headers["User-Agent"], url)
        except Exception:
            return True

    def _response_text(self, response: requests.Response, content_type: str) -> str:
        """Decode HTTP text using the page's declared charset before generic fallbacks."""
        raw = response.content
        lowered = (content_type or "").lower()
        if "html" in lowered or "xhtml" in lowered:
            soup = BeautifulSoup(raw, "html.parser")
            encoding = soup.original_encoding or response.encoding or getattr(response, "apparent_encoding", None) or "utf-8"
            try:
                return raw.decode(encoding, errors="replace")
            except (LookupError, TypeError):
                return raw.decode("utf-8", errors="replace")
        declared = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", lowered)
        if declared:
            encoding = declared.group(1)
            try:
                return raw.decode(encoding, errors="replace")
            except (LookupError, TypeError):
                return raw.decode("utf-8", errors="replace")
        # UTF-8 is the deterministic default. A malformed byte is replaced in-place instead of
        # switching the entire document to a different legacy encoding.
        try:
            return raw.decode(str(self.default_text_encoding or "utf-8"), errors="replace")
        except (LookupError, TypeError):
            return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _looks_like_xml(content_type: str) -> bool:
        value = content_type.lower()
        return "xml" in value or "rss" in value or "atom" in value

    @staticmethod
    def _xml_text(xml_text: str) -> str:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml_text)).strip()
        parts = [chunk.strip() for chunk in root.itertext() if chunk and chunk.strip()]
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    @classmethod
    def extract_text(cls, html: str, *, xml: bool = False) -> str:
        if xml:
            return cls._xml_text(html)
        soup = BeautifulSoup(html, "html.parser")
        # Remove non-content regions as well as executable/style content. This is a
        # structural extraction policy: it does not infer semantic facts from a page.
        for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "template", "dialog"]):
            tag.decompose()
        for tag in soup.find_all(attrs={"aria-hidden": re.compile(r"^true$", re.I)}):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": re.compile(r"^(navigation|contentinfo|dialog)$", re.I)}):
            tag.decompose()
        # Conservative structural filtering for common page-chrome widgets.
        # These heuristics clean ingestion; they do not infer factual meaning.
        chrome_pattern = re.compile(
            r"(?:^|[-_\s])(?:cookie|consent|newsletter|subscribe|social|share|"
            r"related|recommend(?:ation)?s?|breadcrumb|pagination|toolbar|"
            r"advert(?:isement)?|ad-banner|site-nav|menu)(?:$|[-_\s])",
            re.I,
        )
        for tag in list(soup.find_all(True)):
            if tag.parent is None or tag.name in {"html", "body", "main", "article"}:
                continue
            metadata = " ".join([
                str(tag.get("id", "")),
                " ".join(str(item) for item in (tag.get("class") or [])),
                str(tag.get("role", "")),
            ])
            if metadata and chrome_pattern.search(metadata):
                if len(tag.get_text(" ", strip=True)) <= 1800:
                    tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return cls._deduplicate_repeated_sentences(cls._remove_boilerplate_phrases(text))

    @staticmethod
    def _remove_boilerplate_phrases(text: str) -> str:
        """Remove standalone common web UI phrases while preserving article prose."""
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if not value:
            return ""
        patterns = (
            r"^(?:skip to (?:main )?content|toggle navigation|open menu|close menu)$",
            r"^(?:accept|manage|reject) (?:all )?(?:cookies|cookie preferences)$",
            r"^(?:sign in|log in|register|subscribe now|read more|learn more)$",
            r"^(?:share|share this (?:article|story|page)|follow us)$",
            r"^(?:advertisement|sponsored content|related articles?)$",
            r"^(?:all rights reserved|privacy policy|terms of (?:service|use))$",
        )
        compiled = [re.compile(pattern, re.I) for pattern in patterns]
        segments = re.split(r"(?<=[.!?])\s+", value)
        kept = []
        for segment in segments:
            candidate = segment.strip().strip("•|")
            normalized = re.sub(r"[.!?]+$", "", candidate).strip()
            if len(normalized) <= 80 and any(pattern.fullmatch(normalized) for pattern in compiled):
                continue
            kept.append(candidate)
        return " ".join(kept)

    @staticmethod
    def _deduplicate_repeated_sentences(text: str) -> str:
        """Remove exact repeated sentence blocks while retaining first-seen order."""
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        output: list[str] = []
        seen: set[str] = set()
        for part in parts:
            key = re.sub(r"\W+", " ", part.casefold()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(part.strip())
        return " ".join(output)

    @staticmethod
    def assess_content_quality(text: str) -> dict:
        """Return structural text-quality diagnostics; never claims factual truth."""
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        tokens = re.findall(r"[\w'-]+", value.casefold(), flags=re.UNICODE)
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        alphanumeric_ratio = sum(char.isalnum() for char in value) / max(1, len(value))
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
        sentence_keys = [re.sub(r"\W+", " ", part.casefold()).strip() for part in sentence_parts]
        repeated_sentence_ratio = 1.0 - (len(set(sentence_keys)) / len(sentence_keys)) if sentence_keys else 0.0
        flags: list[str] = []
        if len(value) < 80:
            flags.append("short_text")
        if len(value) >= 120 and alphanumeric_ratio < 0.25:
            flags.append("low_alphanumeric_density")
        if len(tokens) >= 80 and unique_ratio < 0.10:
            flags.append("very_low_token_diversity")
        if len(sentence_keys) >= 8 and repeated_sentence_ratio > 0.50:
            flags.append("repeated_sentence_content")
        # Keep short-but-informative residual passages usable after a separate
        # evaluation-overlap filter removes contaminated text. The ingestion path has
        # its own 32-character minimum; this diagnostic does not relabel all short prose
        # as boilerplate. Repetition/low-density flags remain hard rejection signals.
        reject = len(value) < 32 or "low_alphanumeric_density" in flags or "very_low_token_diversity" in flags or "repeated_sentence_content" in flags
        return {
            "schema_version": 1,
            "status": "low_quality" if reject else ("short_but_usable" if "short_text" in flags else "usable"),
            "character_count": len(value),
            "token_count": len(tokens),
            "unique_token_ratio": unique_ratio,
            "alphanumeric_ratio": alphanumeric_ratio,
            "repeated_sentence_ratio": repeated_sentence_ratio,
            "flags": flags,
            "limitations": "Structural heuristics only; quality flags do not validate factual correctness or trustworthiness.",
        }

    @classmethod
    def normalize_many(cls, urls: list[str], base_url: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for raw in urls:
            value = cls.normalize_url(raw, base_url)
            if value and value not in seen:
                seen.add(value)
                found.append(value)
        return found

    @classmethod
    def extract_links(cls, html: str, base_url: str, *, xml: bool = False) -> list[str]:
        if xml:
            try:
                root = ET.fromstring(html)
                raw: list[str] = []
                for node in root.iter():
                    tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
                    if tag == "link":
                        href = node.attrib.get("href") or (node.text or "").strip()
                        if href:
                            raw.append(href)
                return cls.normalize_many(raw, base_url)
            except ET.ParseError:
                return []
        soup = BeautifulSoup(html, "html.parser")
        return cls.normalize_many([x.get("href", "") for x in soup.find_all("a", href=True)], base_url)

    @classmethod
    def extract_feed_links(cls, html: str, base_url: str) -> list[str]:
        try:
            root = ET.fromstring(html)
        except ET.ParseError:
            return []
        raw: list[str] = []
        for node in root.iter():
            tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
            if tag == "link":
                href = node.attrib.get("href") or (node.text or "").strip()
                if href:
                    raw.append(href)
        return cls.normalize_many(raw, base_url)

    @staticmethod
    def page_title(html: str, *, xml: bool = False) -> str:
        if xml:
            try:
                root = ET.fromstring(html)
                for node in root.iter():
                    tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
                    if tag == "title" and (node.text or "").strip():
                        return (node.text or "").strip()[:500]
            except ET.ParseError:
                return ""
            return ""
        soup = BeautifulSoup(html, "html.parser")
        return soup.title.get_text(" ", strip=True)[:500] if soup.title else ""

    def open(self, url: str) -> Page:
        normalized = self.normalize_url(url, allowed_schemes=self.config.allowed_schemes)
        if not normalized:
            raise ValueError("URL must be absolute and use an allowed public HTTP(S) scheme")
        host = urllib.parse.urlparse(normalized).hostname or ""
        if not self._host_is_public(host):
            raise ValueError("URL resolves to a non-public or unresolvable host")
        if not self._robots(normalized):
            raise WebAccessError(normalized, "robots.txt disallows this URL")
        response = self._get(normalized)
        response.raise_for_status()
        ctype = response.headers.get("content-type", "").lower()
        is_pdf = "application/pdf" in ctype or urllib.parse.urlparse(response.url).path.lower().endswith(".pdf")
        allowed = ("text/html", "text/plain", "application/xhtml+xml", "application/rss+xml", "application/atom+xml", "xml", "json")
        if not is_pdf and not any(x in ctype for x in allowed):
            raise ValueError(f"unsupported content type: {ctype}")
        if is_pdf:
            text = read_document_bytes(response.content, name=urllib.parse.urlparse(response.url).path or "download.pdf", content_type=ctype)
            title = Path(urllib.parse.urlparse(response.url).path).name or response.url
            bounded_text = text[: self.max_pdf_chars]
            return Page(response.url, title, bounded_text, response.status_code, [], ctype, depth=0, quality_report=self.assess_content_quality(bounded_text))
        content = self._response_text(response, ctype)
        is_xml = self._looks_like_xml(ctype)
        title = self.page_title(content, xml=is_xml)
        text = self.extract_text(content, xml=is_xml)
        links = self.extract_links(content, response.url, xml=is_xml)
        if is_xml:
            for link in self.extract_feed_links(content, response.url):
                if link not in links:
                    links.append(link)
        bounded_text = text[: self.max_chars]
        return Page(response.url, title or response.url, bounded_text, response.status_code, links, ctype, depth=0, quality_report=self.assess_content_quality(bounded_text))

    def crawl_same_domain(self, url: str, *, max_pages: int = 8, max_depth: int = 2, max_attempts: int = 32) -> list[Page]:
        """Bounded same-domain breadth-first crawl starting from an explicit URL."""
        self.last_errors = []
        limit = max(1, int(max_pages))
        depth_limit = max(0, int(max_depth))
        attempt_limit = max(limit, int(max_attempts))
        start = self.normalize_url(url, allowed_schemes=self.config.allowed_schemes)
        if not start:
            raise ValueError("URL must be absolute and use an allowed public HTTP(S) scheme")
        root_host = (urlsplit(start).hostname or "").lower().rstrip(".")
        queue: list[tuple[str, int]] = [(start, 0)]
        queued = {start}
        visited: set[str] = set()
        pages: list[Page] = []
        attempts = 0
        while queue and len(pages) < limit and attempts < attempt_limit:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            attempts += 1
            try:
                page = self.open(current)
            except Exception as exc:
                self.last_errors.append(f"crawl {current}: {exc!r}")
                continue
            page.depth = depth
            pages.append(page)
            if depth >= depth_limit:
                continue
            for link in page.links:
                normalized = self.normalize_url(link, current, self.config.allowed_schemes)
                if not normalized or normalized in queued:
                    continue
                link_host = (urlsplit(normalized).hostname or "").lower().rstrip(".")
                if link_host != root_host:
                    continue
                if any(pattern.search(urllib.parse.urlsplit(normalized).path) for pattern in self.crawl_exclude_patterns):
                    continue
                queued.add(normalized)
                queue.append((normalized, depth + 1))
        if not pages and self.last_errors:
            self.last_errors = self.last_errors[-min(len(self.last_errors), 8):]
        return pages

    def discover(self) -> Page:
        self.last_errors = []
        choices: list[DiscoverySource] = list(self.config.bootstrap_discovery)
        if not choices:
            raise RuntimeError("no discovery sources are configured")
        failures: list[str] = []
        for source in random.sample(choices, len(choices)):
            try:
                page = self.open(source.url)
                if source.type == "feed" and page.links:
                    return self.open(random.choice(page.links))
                return page
            except Exception as exc:
                failures.append(f"{source.name}: {exc!r}")
        self.last_errors = failures
        raise RuntimeError("all configured discovery sources failed: " + " | ".join(failures))

    def search(self, query: str, limit: int = 5) -> list[str]:
        self.last_errors = []
        query = " ".join(query.split())[:200]
        if len(query) < 3:
            return []
        providers = list(self.config.search_providers)
        if not providers:
            return []
        start = self._provider_index % len(providers)
        self._provider_index += 1
        for provider in providers[start:] + providers[:start]:
            try:
                response = self._search_with_provider(provider, query)
                if response:
                    return response[:limit]
            except Exception as exc:
                self.last_errors.append(f"{provider.name}: {exc!r}")
                continue
        return []

    def _search_with_provider(self, provider: WebSearchProvider, query: str) -> list[str]:
        response = self._get(provider.url, params={provider.query_parameter: query})
        response.raise_for_status()
        soup = BeautifulSoup(self._response_text(response, response.headers.get("content-type", "")), "html.parser")
        if not provider.result_selector:
            return []
        urls: list[str] = []
        for node in soup.select(provider.result_selector):
            href = node.get(provider.result_attribute)
            value = self.normalize_url(href, response.url, self.config.allowed_schemes)
            if value and value not in urls:
                urls.append(value)
        return urls

    def close(self) -> None:
        self.session.close()
