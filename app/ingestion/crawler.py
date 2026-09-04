"""Layer 03 - Website crawler.

Polite, same-origin, breadth-first. Reads robots.txt, prefers the sitemap when
one exists, and caps pages/depth/concurrency because a property site with a
booking-engine calendar can otherwise generate an unbounded URL space
(?date=2026-09-01, ?date=2026-09-02, ...).

Extraction uses trafilatura, which strips the nav/footer/cookie-banner
boilerplate that would otherwise be duplicated into every single chunk and
poison retrieval with 300 identical "Book Now · Rooms · Contact" fragments.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from selectolax.parser import HTMLParser

from app.config import Settings
from app.logging_setup import get_logger
from app.models.domain import Document, SourceKind

log = get_logger(__name__)


def extract_main_content(html: str) -> str | None:
    """Pull the readable content out of a page.

    Ingestion and the drift detector MUST both go through this. They hash the
    result and compare, so any difference in trafilatura flags - include_comments
    defaults to True, for instance - makes every page look permanently changed.
    """
    return trafilatura.extract(
        html,
        output_format="markdown",
        include_links=False,
        include_tables=True,
        include_comments=False,
        favor_recall=True,
    )

# Never worth crawling on a hotel site.
_SKIP_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|zip|mp4|mov|avi|woff2?|ttf|eot)(\?|$)",
    re.I,
)
# Booking engines, calendars, logins, and infinite filter permutations.
_SKIP_PATH = re.compile(
    r"/(wp-admin|wp-login|cart|checkout|login|signin|signup|account|search|tag|author)/|"
    r"[?&](date|checkin|check_in|arrival|departure|page|sort|filter)=",
    re.I,
)


@dataclass
class CrawlResult:
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pages_seen: int = 0


class SiteCrawler:
    def __init__(self, settings: Settings, property_id: str) -> None:
        self.settings = settings
        self.property_id = property_id
        self._robots: RobotFileParser | None = None

    async def crawl(
        self,
        start_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> CrawlResult:
        max_pages = max_pages or self.settings.crawl_max_pages
        max_depth = max_depth or self.settings.crawl_max_depth

        origin = urlparse(start_url)
        if origin.scheme not in ("http", "https") or not origin.netloc:
            raise ValueError(f"start_url must be an absolute http(s) URL, got {start_url!r}")

        result = CrawlResult()
        seen: set[str] = set()
        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()

        headers = {"User-Agent": self.settings.crawl_user_agent}
        limits = httpx.Limits(max_connections=self.settings.crawl_concurrency * 2)

        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=20.0,
            limits=limits,
        ) as client:
            if self.settings.crawl_respect_robots:
                await self._load_robots(client, origin)

            # Seeds go through the same gate as discovered links. Skipping
            # it would let a sitemap bypass robots.txt, include_paths and
            # exclude_paths entirely - an operator excluding /blog would still
            # get every blog page the sitemap lists.
            seeds = await self._seed_urls(client, start_url, origin)
            for url in seeds:
                if url in seen:
                    continue
                # The start_url is the operator's explicit instruction, so it is
                # admitted even when the filters would not have discovered it.
                if url != start_url and not self._allowed(
                    url, origin, include_paths, exclude_paths
                ):
                    continue
                seen.add(url)
                queue.put_nowait((url, 0))

            semaphore = asyncio.Semaphore(self.settings.crawl_concurrency)
            lock = asyncio.Lock()

            async def worker() -> None:
                while True:
                    try:
                        url, depth = await asyncio.wait_for(queue.get(), timeout=2.0)
                    except TimeoutError:
                        return
                    try:
                        async with lock:
                            if len(result.documents) >= max_pages:
                                return
                        async with semaphore:
                            await asyncio.sleep(self.settings.crawl_delay_seconds)
                            doc, links = await self._fetch(client, url)

                        async with lock:
                            result.pages_seen += 1
                            if doc is not None and len(result.documents) < max_pages:
                                result.documents.append(doc)
                            if depth < max_depth:
                                for link in links:
                                    if link in seen or len(seen) >= max_pages * 4:
                                        continue
                                    if not self._allowed(
                                        link, origin, include_paths, exclude_paths
                                    ):
                                        continue
                                    seen.add(link)
                                    queue.put_nowait((link, depth + 1))
                    except Exception as exc:  # one bad page must not kill the crawl
                        async with lock:
                            result.errors.append(f"{url}: {type(exc).__name__}: {exc}")
                    finally:
                        queue.task_done()

            workers = [
                asyncio.create_task(worker())
                for _ in range(self.settings.crawl_concurrency)
            ]
            await asyncio.gather(*workers, return_exceptions=True)

        log.info(
            "crawl.done",
            property_id=self.property_id,
            start_url=start_url,
            documents=len(result.documents),
            seen=result.pages_seen,
            errors=len(result.errors),
        )
        return result

    # -- internals -------------------------------------------------------

    async def _load_robots(self, client: httpx.AsyncClient, origin) -> None:
        robots_url = f"{origin.scheme}://{origin.netloc}/robots.txt"
        parser = RobotFileParser()
        try:
            response = await client.get(robots_url)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                # No robots.txt means nothing is disallowed. allow_all is real
                # at runtime; typeshed just does not declare it.
                parser.allow_all = True  # type: ignore[attr-defined]
        except httpx.HTTPError:
            parser.allow_all = True  # type: ignore[attr-defined]
        self._robots = parser

    async def _seed_urls(
        self, client: httpx.AsyncClient, start_url: str, origin
    ) -> list[str]:
        """Prefer the sitemap: it gives full coverage without deep link-walking."""
        urls = [start_url]
        sitemap_url = f"{origin.scheme}://{origin.netloc}/sitemap.xml"
        try:
            response = await client.get(sitemap_url)
            if response.status_code == 200 and "xml" in response.headers.get(
                "content-type", ""
            ):
                found = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)
                same_origin = [
                    u for u in found if urlparse(u).netloc == origin.netloc
                ]
                urls.extend(same_origin[: self.settings.crawl_max_pages])
                log.info("crawl.sitemap", count=len(same_origin), url=sitemap_url)
        except httpx.HTTPError:
            pass
        return list(dict.fromkeys(urls))

    def _allowed(
        self,
        url: str,
        origin,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
    ) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != origin.netloc or parsed.scheme not in ("http", "https"):
            return False
        if _SKIP_EXT.search(url) or _SKIP_PATH.search(url):
            return False
        if exclude_paths and any(p in parsed.path for p in exclude_paths):
            return False
        if include_paths and not any(p in parsed.path for p in include_paths):
            return False
        # Kept as a guard clause to match the chain above, rather than
        # collapsing into one negated return.
        if self._robots is not None and not self._robots.can_fetch(  # noqa: SIM103
            self.settings.crawl_user_agent, url
        ):
            return False
        return True

    async def _fetch(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[Document | None, list[str]]:
        response = await client.get(url)
        if response.status_code != 200:
            return None, []
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type:
            return None, []

        html = response.text
        links = self._extract_links(html, url)

        markdown = extract_main_content(html)
        if not markdown or len(markdown.strip()) < 120:
            return None, links

        doc = Document(
            property_id=self.property_id,
            source_kind=SourceKind.WEBSITE,
            uri=str(response.url),
            title=self._title(html) or urlparse(url).path.strip("/") or "Home",
            text=markdown,
            # Kept so the drift detector can send a conditional request and
            # settle most pages with a 304 instead of a full download.
            metadata={
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
            },
        )
        return doc, links

    @staticmethod
    def _title(html: str) -> str:
        tree = HTMLParser(html)
        for selector in ("h1", "title"):
            node = tree.css_first(selector)
            if node and node.text(strip=True):
                return node.text(strip=True)[:200]
        return ""

    @staticmethod
    def _extract_links(html: str, base_url: str) -> list[str]:
        tree = HTMLParser(html)
        out: list[str] = []
        for node in tree.css("a[href]"):
            href = node.attributes.get("href")
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute, _ = urldefrag(urljoin(base_url, href))
            out.append(absolute)
        return list(dict.fromkeys(out))
