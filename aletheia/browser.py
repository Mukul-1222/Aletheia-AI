from __future__ import annotations

import base64
import io
import json
import re
import shutil
import threading
import time
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class BrowserEnvironmentError(RuntimeError):
    pass


@dataclass
class BrowserElement:
    ref: str
    tag: str
    role: str
    text: str
    aria_label: str
    name: str
    placeholder: str
    input_type: str
    href: str = ""
    disabled: bool = False
    signature: str = ""


@dataclass
class BrowserObservation:
    url: str
    title: str
    text: str
    elements: list[BrowserElement]
    screenshot_b64: str = ""
    timestamp: float = 0.0
    navigation_id: int = 0
    observation_id: int = 0

    def compact(self, *, max_text: int = 5000, max_elements: int = 80) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        body = re.sub(r"\s+", " ", self.text).strip()
        if body:
            lines.append(f"TEXT: {body[:max(0, int(max_text))]}")
        for item in self.elements[:max(0, int(max_elements))]:
            attrs = []
            if item.role:
                attrs.append(f"role={item.role}")
            if item.aria_label:
                attrs.append(f"aria={item.aria_label}")
            if item.name:
                attrs.append(f"name={item.name}")
            if item.placeholder:
                attrs.append(f"placeholder={item.placeholder}")
            if item.input_type:
                attrs.append(f"type={item.input_type}")
            if item.href:
                attrs.append(f"href={item.href}")
            suffix = f" [{', '.join(attrs)}]" if attrs else ""
            label = item.text or item.aria_label or item.name or item.placeholder
            lines.append(f"[{item.ref}] <{item.tag}>{label[:240]}{suffix}")
        return "\n".join(lines)

    def to_dict(self, include_screenshot: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_screenshot:
            payload.pop("screenshot_b64", None)
        return payload


class BrowserEnvironment:
    """Interactive browser environment using a local Chromium process.

    The environment contains only generic browser mechanics. It has no built-in
    knowledge of sites, objects, people, task semantics, or success rules. Aletheia
    receives observations and selects generic browser actions from her learned state.
    """

    _interactive_script = r"""
    (() => {
      const selectors = [
        'a[href]', 'button', 'input', 'textarea', 'select', 'option',
        '[role="button"]', '[role="link"]', '[role="textbox"]',
        '[role="combobox"]', '[contenteditable="true"]'
      ].join(',');
      const nodes = Array.from(document.querySelectorAll(selectors));
      const visible = [];
      let index = 0;
      for (const node of nodes) {
        const style = window.getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        const visibleNow = style.visibility !== 'hidden' && style.display !== 'none' &&
          rect.width > 0 && rect.height > 0;
        if (!visibleNow || node.disabled) continue;
        const ref = `e${++index}`;
        node.setAttribute('data-aletheia-ref', ref);
        visible.push({
          ref,
          tag: node.tagName.toLowerCase(),
          role: node.getAttribute('role') || '',
          text: (node.innerText || node.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 500),
          aria_label: node.getAttribute('aria-label') || '',
          name: node.getAttribute('name') || '',
          placeholder: node.getAttribute('placeholder') || '',
          input_type: node.getAttribute('type') || '',
          href: node.href || '',
          disabled: !!node.disabled,
          signature: JSON.stringify({
            tag: node.tagName.toLowerCase(), role: node.getAttribute('role') || '',
            text: (node.innerText || node.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 200),
            aria: node.getAttribute('aria-label') || '', name: node.getAttribute('name') || '',
            placeholder: node.getAttribute('placeholder') || '', type: node.getAttribute('type') || '', href: node.href || ''
          }),
        });
      }
      return visible;
    })()
    """

    def __init__(
        self,
        root: Path,
        web,
        *,
        headless: bool = True,
        executable_path: str = "",
        user_data_dir: str | None = None,
        timeout_ms: int = 20_000,
        respect_robots: bool = True,
        screenshot_max_bytes: int = 8 * 1024 * 1024,
        persistent_profile: bool = False,
        allow_domains: list[str] | None = None,
        allow_form_submit: bool = False,
        allow_downloads: bool = False,
        allow_file_uploads: bool = False,
        allow_authentication: bool = False,
        dns_cache_seconds: float = 30.0,
        egress_proxy_url: str = "",
        egress_proxy_policy_confirmed: bool = False,
    ):
        self.root = Path(root).resolve()
        self.web = web
        self.headless = bool(headless)
        self.executable_path = str(executable_path or "")
        self.timeout_ms = max(1000, int(timeout_ms))
        self.respect_robots = bool(respect_robots)
        self.screenshot_max_bytes = max(64 * 1024, int(screenshot_max_bytes))
        self.persistent_profile = bool(persistent_profile)
        self.allow_domains = {str(item).lower().strip().lstrip(".") for item in (allow_domains or []) if str(item).strip()}
        self.allow_form_submit = bool(allow_form_submit)
        self.allow_downloads = bool(allow_downloads)
        self.allow_file_uploads = bool(allow_file_uploads)
        self.allow_authentication = bool(allow_authentication)
        self.dns_cache_seconds = max(0.0, float(dns_cache_seconds))
        self.egress_proxy_url = str(egress_proxy_url or "").strip()
        # This is an operator attestation, not automatic proof: the proxy must enforce
        # destination-IP policy and the host firewall must prevent Chromium bypass.
        self.egress_proxy_policy_confirmed = bool(egress_proxy_policy_confirmed)
        self.profile_dir = None
        self._ephemeral_profile_dir = None
        if self.persistent_profile:
            if user_data_dir:
                profile = Path(user_data_dir).expanduser()
                self.profile_dir = (profile if profile.is_absolute() else self.root / profile).resolve()
            else:
                self.profile_dir = self.root / "data" / "browser_profile"
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._context = None
        self.page = None
        self._lock = threading.RLock()
        self._navigation_id = 0
        self._observation_id = 0
        self._last_observation: BrowserObservation | None = None
        self._started = False

    @staticmethod
    def _safe_executable(path: str) -> str | None:
        if path:
            candidate = Path(path).expanduser()
            if candidate.exists() and candidate.is_file():
                return str(candidate)
            resolved = shutil.which(path)
            if resolved:
                return resolved
            return None
        for candidate in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    def _external_network_ready(self) -> bool:
        """External browsing requires a configured, policy-enforcing egress proxy.

        Playwright routing alone cannot pin the actual Chromium socket destination, so
        Python-side DNS checks are not treated as a complete DNS-rebinding defense.
        """
        if not self.egress_proxy_url or not self.egress_proxy_policy_confirmed:
            return False
        parsed = urlparse(self.egress_proxy_url)
        return parsed.scheme in {"http", "https", "socks5"} and bool(parsed.hostname and parsed.port)

    def _ensure_started(self) -> None:
        if self._started and self.page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserEnvironmentError(
                "browser support requires the playwright package and a Chromium browser"
            ) from exc
        self._playwright = sync_playwright().start()
        executable = self._safe_executable(self.executable_path)
        proxy_settings = None
        if self._external_network_ready():
            # Empty bypass list: don't let Chromium silently bypass the configured proxy.
            proxy_settings = {"server": self.egress_proxy_url, "bypass": ""}
        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "args": ["--disable-dev-shm-usage", "--disable-dns-prefetch", "--disable-features=AsyncDns,PrefetchPrivacyChanges,PreconnectToSearch"],
            "accept_downloads": bool(self.allow_downloads),
            # A service worker can create network paths outside ordinary page routing.
            "service_workers": "block",
        }
        if executable:
            launch_kwargs["executable_path"] = executable
        if proxy_settings is not None:
            launch_kwargs["proxy"] = proxy_settings
        try:
            if self.persistent_profile:
                self._context = self._playwright.chromium.launch_persistent_context(
                    str(self.profile_dir), **launch_kwargs,
                )
            else:
                launch_kwargs = {
                    "headless": self.headless,
                    "executable_path": executable or None,
                    "args": ["--disable-dev-shm-usage", "--disable-dns-prefetch", "--disable-features=AsyncDns,PrefetchPrivacyChanges,PreconnectToSearch"],
                }
                if proxy_settings is not None:
                    launch_kwargs["proxy"] = proxy_settings
                self._playwright_browser = self._playwright.chromium.launch(**launch_kwargs)
                self._context = self._playwright_browser.new_context(accept_downloads=bool(self.allow_downloads), service_workers="block")
                self._ephemeral_profile_dir = Path(tempfile.mkdtemp(prefix="aletheia_browser_"))
        except Exception as exc:
            self.close()
            raise BrowserEnvironmentError(
                "unable to launch Chromium; install Chromium/Playwright or set browser executable_path"
            ) from exc
        self._context.set_default_timeout(self.timeout_ms)
        self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._started = True
        try:
            self._context.route("**/*", self._guard_request)
            websocket_router = getattr(self._context, "route_web_socket", None)
            if not callable(websocket_router):
                raise RuntimeError("Playwright does not provide WebSocket routing; refusing an unguarded browser")
            websocket_router("**/*", self._guard_websocket)
        except Exception as exc:
            self.close()
            raise BrowserEnvironmentError(f"unable to install complete browser network guards: {exc!r}") from exc
        self._attach_page(self.page)

    @staticmethod
    def _guard_websocket(websocket_route) -> None:
        """Block WebSocket traffic unless a future policy explicitly supports it.

        HTTP route guards do not cover every browser networking primitive. Failing
        closed here is safer than silently allowing a page to open an unfiltered socket.
        """
        websocket_route.close(code=1008, reason="WebSocket connections are disabled by Aletheia browser policy")

    def _domain_allowed(self, host: str) -> bool:
        normalized = str(host or "").lower().rstrip(".")
        # An empty allowlist must not mean unrestricted access.
        if not normalized or not self.allow_domains:
            return False
        return any(normalized == domain or normalized.endswith("." + domain) for domain in self.allow_domains)

    def _authentication_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return bool(re.search(r"/(login|signin|signup|register|logout)(?:/|$)", path))

    def _guard_request(self, route) -> None:
        request = route.request
        parsed = urlparse(str(request.url or ""))
        # Browser pages can issue POST/fetch requests through JavaScript without
        # clicking a detectable HTML submit button. Unless form submission is
        # explicitly enabled, permit only read-only navigation/resource methods.
        method = str(getattr(request, "method", "GET") or "GET").upper()
        if method not in {"GET", "HEAD"} and not self.allow_form_submit:
            route.abort()
            return
        if parsed.scheme in {"data", "blob"}:
            route.continue_()
            return
        if parsed.scheme not in {"http", "https"}:
            route.abort()
            return
        # No direct remote browser requests: route-based hostname checks do not pin
        # Chromium's actual socket destination and cannot alone prevent DNS rebinding.
        if not self._external_network_ready():
            route.abort()
            return
        host = parsed.hostname or ""
        # Do not reuse the web client's DNS cache for browser request checks. Chromium
        # resolves independently, so this is defense-in-depth rather than full pinning.
        cache = getattr(self.web, "_public_host_cache", None)
        if isinstance(cache, dict):
            cache.pop(host.strip("[]").lower().rstrip("."), None)
        if not self._domain_allowed(host) or not self.web._host_is_public(host):
            route.abort()
            return
        if not self.allow_authentication and self._authentication_url(str(request.url)):
            route.abort()
            return
        route.continue_()

    def _attach_page(self, page) -> None:
        self.page = page
        try:
            page.on("framenavigated", lambda frame: self._on_navigation(frame.url) if frame == page.main_frame else None)
        except Exception:
            pass

    def _on_navigation(self, url: str) -> None:
        self._navigation_id += 1

    def _validate_url(self, url: str, *, check_robots: bool | None = None) -> str:
        normalized = self.web.normalize_url(url, allowed_schemes=self.web.config.allowed_schemes)
        if not normalized:
            raise BrowserEnvironmentError("browser URLs must use an allowed public HTTP(S) scheme")
        host = urlparse(normalized).hostname or ""
        if not self._external_network_ready():
            raise BrowserEnvironmentError(
                "external browser navigation is disabled: configure a policy-enforcing egress proxy and confirm host-level no-bypass enforcement"
            )
        if not self.allow_domains:
            raise BrowserEnvironmentError("external browser navigation requires a non-empty domain allowlist")
        if not self._domain_allowed(host):
            raise BrowserEnvironmentError("browser target is outside the configured domain allowlist")
        if not self.web._host_is_public(host):
            raise BrowserEnvironmentError("browser target resolves to a non-public or unresolvable host")
        if not self.allow_authentication and self._authentication_url(normalized):
            raise BrowserEnvironmentError("browser authentication routes are blocked")
        obey = self.respect_robots if check_robots is None else bool(check_robots)
        if obey and not self.web._robots(normalized):
            raise BrowserEnvironmentError("robots.txt disallows this browser target")
        return normalized

    def navigate(self, url: str) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            normalized = self._validate_url(url)
            self.page.goto(normalized, wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._validate_current_location()
            return self.observe(screenshot=True)

    def _validate_current_location(self) -> None:
        current = str(self.page.url or "")
        if not current or current == "about:blank":
            return
        self._validate_url(current, check_robots=False)

    def _refresh_refs(self) -> list[dict[str, Any]]:
        return self.page.evaluate(self._interactive_script)

    def observe(self, *, screenshot: bool = True) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self._validate_current_location()
            elements_raw = self._refresh_refs()
            elements = [BrowserElement(**item) for item in elements_raw]
            text = self.page.locator("body").inner_text(timeout=self.timeout_ms) if self.page.locator("body").count() else ""
            text = re.sub(r"\s+", " ", text).strip()
            shot = ""
            if screenshot:
                data = self.page.screenshot(type="png", full_page=False)
                if len(data) <= self.screenshot_max_bytes:
                    shot = base64.b64encode(data).decode("ascii")
            self._observation_id += 1
            obs = BrowserObservation(
                url=str(self.page.url),
                title=str(self.page.title())[:500],
                text=text,
                elements=elements,
                screenshot_b64=shot,
                timestamp=time.time(),
                navigation_id=self._navigation_id,
                observation_id=self._observation_id,
            )
            self._last_observation = obs
            return obs

    def screenshot_bytes(self, *, full_page: bool = False) -> bytes:
        with self._lock:
            self._ensure_started()
            data = self.page.screenshot(type="png", full_page=bool(full_page))
            if len(data) > self.screenshot_max_bytes:
                raise BrowserEnvironmentError("browser screenshot exceeds configured size limit")
            return data

    def _locator(self, ref: str, *, observation_id: int | None = None, signature: str | None = None):
        value = str(ref or "").strip()
        if not re.fullmatch(r"e\d+", value):
            raise BrowserEnvironmentError("invalid browser element reference")
        if self._last_observation is None:
            raise BrowserEnvironmentError("no browser observation exists")
        if observation_id is not None and int(observation_id) != int(self._last_observation.observation_id):
            raise BrowserEnvironmentError("browser observation is stale")
        expected = {item.ref: item for item in self._last_observation.elements}.get(value)
        if expected is None:
            raise BrowserEnvironmentError("browser element reference is stale or not present")
        locator = self.page.locator(f'[data-aletheia-ref="{value}"]')
        if locator.count() != 1:
            raise BrowserEnvironmentError("browser element reference is stale or not present")
        live = self.page.evaluate("""ref => { const n=document.querySelector(`[data-aletheia-ref=\"${ref}\"]`); if(!n) return null; return JSON.stringify({tag:n.tagName.toLowerCase(),role:n.getAttribute('role')||'',text:(n.innerText||n.textContent||'').trim().replace(/\\s+/g,' ').slice(0,200),aria:n.getAttribute('aria-label')||'',name:n.getAttribute('name')||'',placeholder:n.getAttribute('placeholder')||'',type:n.getAttribute('type')||'',href:n.href||''}); }""", value)
        expected_sig = str(signature or expected.signature or "")
        if live is None or (expected_sig and str(live) != expected_sig):
            raise BrowserEnvironmentError("browser element reference changed since the observed page state")
        return locator, expected

    def click(self, ref: str, *, observation_id: int | None = None, signature: str | None = None) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            locator, element = self._locator(ref, observation_id=observation_id, signature=signature)
            if str(element.input_type).lower() == "file" and not self.allow_file_uploads:
                raise BrowserEnvironmentError("file uploads are disabled")
            is_submit = locator.evaluate("el => { const t=(el.getAttribute('type')||'').toLowerCase(); return t==='submit' || t==='image' || !!el.closest('form') && (el.tagName==='BUTTON' && (t==='' || t==='submit')); }")
            if bool(is_submit) and not self.allow_form_submit:
                raise BrowserEnvironmentError("form submission is disabled")
            before = self.page.url
            locator.click(timeout=self.timeout_ms)
            self._settle(before)
            return self.observe(screenshot=True)

    def type_text(self, ref: str, text: str, *, submit: bool = False, observation_id: int | None = None, signature: str | None = None, approve_sensitive: bool = False) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            locator, _ = self._locator(ref, observation_id=observation_id, signature=signature)
            input_type = str(locator.get_attribute("type") or "").lower()
            if input_type == "file" and not self.allow_file_uploads:
                raise BrowserEnvironmentError("file uploads are disabled")
            if input_type == "password" and (not self.allow_authentication or not approve_sensitive):
                raise BrowserEnvironmentError("typing into password fields is disabled")
            value = str(text)
            locator.fill(value)
            if submit:
                if not self.allow_form_submit:
                    raise BrowserEnvironmentError("form submission is disabled")
                locator.press("Enter")
                self._settle(self.page.url)
            return self.observe(screenshot=True)

    def select(self, ref: str, value: str, *, observation_id: int | None = None, signature: str | None = None) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            locator, _ = self._locator(ref, observation_id=observation_id, signature=signature)
            tag = str(locator.evaluate("el => el.tagName.toLowerCase()"))
            if tag != "select":
                raise BrowserEnvironmentError("select action requires a select element")
            locator.select_option(label=str(value))
            return self.observe(screenshot=True)

    def press(self, ref: str, key: str = "Enter", *, observation_id: int | None = None, signature: str | None = None) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            locator, _ = self._locator(ref, observation_id=observation_id, signature=signature)
            if str(key).lower() in {"enter", "numpadenter"} and not self.allow_form_submit:
                inside_form = bool(locator.evaluate("el => !!el.closest('form')"))
                if inside_form:
                    raise BrowserEnvironmentError("form submission is disabled")
            locator.press(str(key))
            self._settle(self.page.url)
            return self.observe(screenshot=True)

    def scroll(self, delta_y: int = 700) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self.page.mouse.wheel(0, int(delta_y))
            self.page.wait_for_timeout(min(500, max(0, self.timeout_ms // 20)))
            return self.observe(screenshot=True)

    def back(self) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self.page.go_back(wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._validate_current_location()
            return self.observe(screenshot=True)

    def forward(self) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self.page.go_forward(wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._validate_current_location()
            return self.observe(screenshot=True)

    def reload(self) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self.page.reload(wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._validate_current_location()
            return self.observe(screenshot=True)

    def wait(self, milliseconds: int = 500) -> BrowserObservation:
        with self._lock:
            self._ensure_started()
            self.page.wait_for_timeout(max(0, min(int(milliseconds), self.timeout_ms)))
            return self.observe(screenshot=True)

    def action(self, action: dict[str, Any]) -> BrowserObservation:
        kind = str(action.get("action", "observe")).strip().lower()
        if kind == "navigate":
            return self.navigate(str(action.get("url", "")))
        if kind == "observe":
            return self.observe(screenshot=True)
        obs_id = action.get("observation_id")
        signature = action.get("signature")
        if kind == "click":
            return self.click(str(action.get("ref", "")), observation_id=int(obs_id) if obs_id is not None else None, signature=str(signature) if signature else None)
        if kind in {"type", "fill"}:
            return self.type_text(str(action.get("ref", "")), str(action.get("text", "")), submit=bool(action.get("submit", False)), observation_id=int(obs_id) if obs_id is not None else None, signature=str(signature) if signature else None, approve_sensitive=bool(action.get("approve_sensitive", False)))
        if kind == "press":
            return self.press(str(action.get("ref", "")), str(action.get("key", "Enter")), observation_id=int(obs_id) if obs_id is not None else None, signature=str(signature) if signature else None)
        if kind == "select":
            return self.select(str(action.get("ref", "")), str(action.get("value", "")), observation_id=int(obs_id) if obs_id is not None else None, signature=str(signature) if signature else None)
        if kind == "scroll":
            return self.scroll(int(action.get("delta_y", 700)))
        if kind == "back":
            return self.back()
        if kind == "forward":
            return self.forward()
        if kind == "reload":
            return self.reload()
        if kind == "wait":
            return self.wait(int(action.get("milliseconds", 500)))
        raise BrowserEnvironmentError(f"unsupported browser action: {kind}")

    def _settle(self, previous_url: str) -> None:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=min(self.timeout_ms, 5000))
        except Exception:
            pass
        if self.page.url != previous_url:
            self._validate_current_location()
        try:
            self.page.wait_for_timeout(100)
        except Exception:
            pass

    @property
    def last_observation(self) -> BrowserObservation | None:
        return self._last_observation

    def state_summary(self) -> dict[str, Any]:
        obs = self._last_observation
        return {
            "started": bool(self._started),
            "url": obs.url if obs else "",
            "title": obs.title if obs else "",
            "elements": len(obs.elements) if obs else 0,
            "navigation_id": obs.navigation_id if obs else self._navigation_id,
            "has_screenshot": bool(obs and obs.screenshot_b64),
        }

    def close(self) -> None:
        with self._lock:
            try:
                if self._context is not None:
                    self._context.close()
            except Exception:
                pass
            try:
                if self._playwright is not None:
                    self._playwright.stop()
            except Exception:
                pass
            self._context = None
            browser = getattr(self, "_playwright_browser", None)
            try:
                if browser is not None:
                    browser.close()
            except Exception:
                pass
            self._playwright_browser = None
            if self._ephemeral_profile_dir is not None:
                shutil.rmtree(self._ephemeral_profile_dir, ignore_errors=True)
                self._ephemeral_profile_dir = None
            self._playwright = None
            self.page = None
            self._started = False

    def __enter__(self):
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
