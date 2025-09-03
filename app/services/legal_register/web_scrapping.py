import asyncio
import logging
import os
import re
import shutil
import string
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import fitz  # PyMuPDF
from aiohttp import ClientConnectorError, ClientResponseError
from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    Frame,
    Page,
    Browser,
    TimeoutError as PlaywrightTimeoutError,
)

# Configure module logger (caller may configure handlers)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BOPdfScraper:
    """
    Production-grade scraper class that:
      - Navigates to BASE_URL using Playwright
      - Finds the content frame (if any) and extracts PDF links
      - Paginates using robust heuristics
      - Downloads PDFs concurrently using aiohttp
      - Optionally extracts a short text preview from downloaded PDFs using PyMuPDF (fitz)
    """

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    )

    def __init__(
        self,
        base_url: str = "https://www.sgg.gov.ma/BulletinOfficiel.aspx",
        output_dir: str = "pdfs",
        headless: bool = True,
        max_pages: int = 3,
        browser_launch_args: Optional[List[str]] = None,
        download_concurrency: int = 4,
        http_timeout: int = 300,
        max_download_retries: int = 3,
        pdf_preview_chars: int = 1000,
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.max_pages = max_pages
        self.browser_launch_args = browser_launch_args or ["--no-sandbox", "--disable-dev-shm-usage"]
        self.download_concurrency = max(1, int(download_concurrency))
        self.http_timeout = int(http_timeout)
        self.max_download_retries = max(0, int(max_download_retries))
        self.pdf_preview_chars = int(pdf_preview_chars)

        # internal state
        self._session: Optional[aiohttp.ClientSession] = None

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _safe_filename(url: str) -> str:
        """Create a safe filename from a URL (keeps extension)."""
        parsed = urlparse(url)
        name = Path(parsed.path).name or "downloaded.pdf"
        # sanitize
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        cleaned = "".join(c for c in name if c in valid_chars)
        cleaned = cleaned.strip()[:200] or "file.pdf"
        return cleaned

    @staticmethod
    def _is_pdf_link(href: str) -> bool:
        """Quick heuristic whether an href points to a PDF."""
        if not href:
            return False
        href_lower = href.split("?")[0].lower()
        return href_lower.endswith(".pdf")

    @staticmethod
    def _normalize_link(base: str, href: str) -> str:
        """Make an absolute URL and strip tracking fragments where safe."""
        return urljoin(base, href)

    # -------------------------
    # Playwright helpers
    # -------------------------
    async def _find_content_frame(self, page: Page) -> Frame:
        """
        Try multiple heuristics to find a logical content frame.
        Falls back to page.main_frame.
        """
        # Try frames whose URL contains a case-insensitive fragment of the base path
        parsed_base = urlparse(self.base_url)
        base_path_keyword = Path(parsed_base.path).stem.lower() if parsed_base.path else ""
        for f in page.frames:
            try:
                if not f.url:
                    continue
                url_l = f.url.lower()
                if base_path_keyword and base_path_keyword in url_l:
                    logger.debug("Found frame by base keyword: %s", f.url)
                    return f
                # sometimes path contains BulletinOfficiel as in the provided example
                if "bulletinofficiel" in url_l or "bulletin" in url_l:
                    logger.debug("Found frame by 'bulletin' in URL: %s", f.url)
                    return f
            except Exception:
                continue

        # If none found, return main frame
        logger.debug("Content frame not found, using main frame.")
        return page.main_frame

    async def _extract_pdfs_from_frame(self, frame: Frame) -> Set[str]:
        """
        Extract PDF links from a Playwright Frame by parsing frame.content() with BeautifulSoup.
        Returns a set of normalized absolute URLs.
        """
        try:
            content = await frame.content()
        except PlaywrightTimeoutError:
            logger.warning("Timeout reading frame content; returning empty set.")
            return set()
        except Exception as exc:
            logger.exception("Error reading frame content: %s", exc)
            return set()

        soup = BeautifulSoup(content, "html.parser")
        links: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            # Some links might be javascript: or anchors - skip those
            if href.lower().startswith("javascript:") or href.startswith("#"):
                continue
            if self._is_pdf_link(href) or ".pdf" in href.lower():
                full = self._normalize_link(self.base_url, href)
                links.add(full)
        return links

    async def _find_next_button(self, frame: Frame) -> Optional[Frame]:
        """
        Heuristics for finding a 'next' control inside the frame:
        - anchor text containing 'Suivant', 'Next', '>', '›', '»'
        - rel="next"
        - aria-label containing 'next' or 'suivant'
        - CSS class name containing 'next' or 'pagination-next'
        Returns a Playwright element handle or None.
        """
        try:
            # Prefer anchors with rel="next"
            handle = await frame.query_selector('a[rel="next"]')
            if handle:
                return handle

            # aria-label or title heuristics
            candidates = await frame.query_selector_all("a, button")
            for el in candidates:
                try:
                    txt = (await el.text_content() or "").strip()
                    title = (await el.get_attribute("title") or "").strip()
                    aria = (await el.get_attribute("aria-label") or "").strip()
                    combined = f"{txt} {title} {aria}".lower()
                    if any(key in combined for key in ("suivant", "next", "›", "»", ">", "suiv")):
                        return el
                    # class based
                    cls = (await el.get_attribute("class") or "").lower()
                    if "next" in cls or "pagination-next" in cls or "pager-next" in cls:
                        return el
                except Exception:
                    continue
        except Exception:
            logger.debug("Error while searching for next button", exc_info=True)
        return None

    # -------------------------
    # Core scraping: extract links + paginate
    # -------------------------
    # async def scrape_pdf_links(self, max_pages: Optional[int] = None) -> List[str]:
    #     """
    #     Main entry to scrape PDF links from the base_url.
    #     :param max_pages: override the instance max_pages if provided
    #     :return: list of unique PDF URLs
    #     """
    #     max_pages = int(max_pages) if max_pages is not None else self.max_pages
    #     pdf_links: Set[str] = set()

    #     playwright = await async_playwright().__aenter__()
    #     browser: Optional[Browser] = None
    #     try:
    #         browser = await playwright.chromium.launch(headless=self.headless, args=self.browser_launch_args)
    #         page = await browser.new_page(user_agent=self.DEFAULT_USER_AGENT)
    #         await page.goto(self.base_url, wait_until="networkidle", timeout=60000)

    #         frame = await self._find_content_frame(page)

    #         current = 1
    #         prev_links: Set[str] = set()
    #         while current <= max_pages:
    #             logger.info("Scraping page %d/%d", current, max_pages)
    #             try:
    #                 new_links = await self._extract_pdfs_from_frame(frame)
    #             except Exception as exc:
    #                 logger.exception("Failed extracting PDF links on page %d: %s", current, exc)
    #                 new_links = set()

    #             # normalize and deduplicate
    #             normalized = {self._normalize_link(self.base_url, l) for l in new_links}
    #             added = normalized - pdf_links
    #             if added:
    #                 logger.info("Found %d new PDF(s) on page %d", len(added), current)
    #             pdf_links.update(normalized)

    #             # if nothing changed vs prev page, break to avoid infinite loop
    #             if normalized and normalized == prev_links:
    #                 logger.info("Duplicate content detected between pages - stopping pagination.")
    #                 break
    #             prev_links = normalized

    #             # find next button
    #             next_btn = await self._find_next_button(frame)
    #             if not next_btn:
    #                 logger.info("No next button found - stopping after page %d", current)
    #                 break

    #             try:
    #                 # attempt click & wait for navigation/idle
    #                 await next_btn.click()
    #                 # give time for dynamic content to load
    #                 await page.wait_for_load_state("networkidle", timeout=30000)
    #                 # re-resolve content frame (some sites replace frames)
    #                 frame = await self._find_content_frame(page)
    #             except PlaywrightTimeoutError:
    #                 logger.warning("Timeout after clicking next on page %d, trying short sleep", current)
    #                 await asyncio.sleep(2)
    #             except Exception as exc:
    #                 logger.exception("Error clicking next button on page %d: %s", current, exc)
    #                 break

    #             current += 1

    #         logger.info("Scraping finished: %d unique PDF links found", len(pdf_links))
    #         return sorted(pdf_links)
    #     finally:
    #         try:
    #             if browser:
    #                 await browser.close()
    #         finally:
    #             await playwright.__aexit__(None, None, None)

    async def scrape_pdf_links(self, max_pages: Optional[int] = None) -> List[str]:
        max_pages = int(max_pages) if max_pages is not None else self.max_pages
        pdf_links: Set[str] = set()

        # Proper async context manager usage
        async with async_playwright() as playwright:
            browser: Optional[Browser] = None
            try:
                browser = await playwright.chromium.launch(headless=self.headless, args=self.browser_launch_args)
                page = await browser.new_page(user_agent=self.DEFAULT_USER_AGENT)
                await page.goto(self.base_url, wait_until="networkidle", timeout=60000)

                frame = await self._find_content_frame(page)

                current = 1
                prev_links: Set[str] = set()
                while current <= max_pages:
                    logger.info("Scraping page %d/%d", current, max_pages)
                    try:
                        new_links = await self._extract_pdfs_from_frame(frame)
                    except Exception as exc:
                        logger.exception("Failed extracting PDF links on page %d: %s", current, exc)
                        new_links = set()

                    normalized = {self._normalize_link(self.base_url, l) for l in new_links}
                    added = normalized - pdf_links
                    if added:
                        logger.info("Found %d new PDF(s) on page %d", len(added), current)
                    pdf_links.update(normalized)

                    if normalized and normalized == prev_links:
                        logger.info("Duplicate content detected between pages - stopping pagination.")
                        break
                    prev_links = normalized

                    next_btn = await self._find_next_button(frame)
                    if not next_btn:
                        logger.info("No next button found - stopping after page %d", current)
                        break

                    try:
                        await next_btn.click()
                        await page.wait_for_load_state("networkidle", timeout=30000)
                        frame = await self._find_content_frame(page)
                    except Exception as exc:
                        logger.exception("Error clicking next button on page %d: %s", current, exc)
                        break

                    current += 1

                logger.info("Scraping finished: %d unique PDF links found", len(pdf_links))
                return sorted(pdf_links)
            finally:
                if browser:
                    await browser.close()


    # -------------------------
    # Downloads (async)
    # -------------------------
    async def _ensure_session(self):
        if self._session and not self._session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=self.http_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": self.DEFAULT_USER_AGENT})

    async def _download_one(self, url: str, dest: Path) -> Tuple[str, Path, int]:
        """
        Download one URL to dest path with retries.
        Returns tuple (url, path, status_code). Raises on persistent failures.
        """
        await self._ensure_session()
        assert self._session is not None  # for typing

        attempt = 0
        last_exc = None
        while attempt <= self.max_download_retries:
            try:
                async with self._session.get(url, timeout=self.http_timeout) as resp:
                    resp.raise_for_status()
                    # stream to file
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with tmp.open("wb") as fh:
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            if not chunk:
                                break
                            fh.write(chunk)
                    tmp.replace(dest)
                    return (url, dest, resp.status)
            except (ClientConnectorError, ClientResponseError, asyncio.TimeoutError) as exc:
                logger.warning("Download attempt %d failed for %s: %s", attempt + 1, url, exc)
                last_exc = exc
                attempt += 1
                await asyncio.sleep(1.5 ** attempt)
            except Exception as exc:
                logger.exception("Unexpected error while downloading %s: %s", url, exc)
                last_exc = exc
                attempt += 1
                await asyncio.sleep(1.5 ** attempt)
        # All retries exhausted
        raise RuntimeError(f"Failed to download {url} after {self.max_download_retries} retries") from last_exc

    async def download_pdfs(self, urls: Iterable[str], overwrite: bool = False) -> List[Dict]:
        """
        Download all URLs concurrently, save into output_dir.
        Returns list of metadata dicts:
           {url, filename, path, status, status_code, error (optional)}
        """
        await self._ensure_session()
        sem = asyncio.Semaphore(self.download_concurrency)
        results: List[Dict] = []

        async def _worker(u: str):
            async with sem:
                fname = self._safe_filename(u)
                dest = self.output_dir / fname
                if dest.exists() and not overwrite:
                    logger.info("Skipping existing file: %s", dest)
                    return {"url": u, "filename": fname, "path": str(dest), "status": "skipped", "status_code": 200}
                try:
                    url, path, code = await self._download_one(u, dest)
                    logger.info("Downloaded %s -> %s", u, path)
                    return {"url": u, "filename": fname, "path": str(path), "status": "ok", "status_code": code}
                except Exception as exc:
                    logger.exception("Failed to download %s", u)
                    return {"url": u, "filename": fname, "path": str(dest), "status": "error", "error": str(exc)}

        tasks = [asyncio.create_task(_worker(u)) for u in urls]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            results.append(res)

        # cleanup session optionally kept open for subsequent calls
        return results

    # -------------------------
    # PDF preview (synchronous heavy op executed in thread)
    # -------------------------
    async def extract_pdf_preview(self, path: str, max_chars: Optional[int] = None) -> str:
        """
        Extract textual preview from a local PDF file using PyMuPDF.
        Runs in a thread to avoid blocking the event loop.
        """
        max_chars = max_chars or self.pdf_preview_chars

        def _open_and_extract(p: str, n: int) -> str:
            try:
                doc = fitz.open(p)
                text_parts = []
                for i, pg in enumerate(doc):
                    try:
                        text_parts.append(pg.get_text())
                    except Exception:
                        continue
                    # limit pages to avoid huge extraction
                    if i >= 4 and sum(len(s) for s in text_parts) > n:
                        break
                doc.close()
                combined = "\n".join(text_parts)
                # normalize whitespace
                combined = re.sub(r"\s+", " ", combined).strip()
                return combined[:n]
            except Exception as exc:
                logger.exception("Error extracting preview from %s: %s", p, exc)
                return ""

        return await asyncio.to_thread(_open_and_extract, path, max_chars)

    # -------------------------
    # High-level convenience runner
    # -------------------------
    async def run(
        self,
        max_pages: Optional[int] = None,
        download: bool = True,
        overwrite: bool = False,
        extract_preview: bool = True,
    ) -> Dict[str, object]:
        """
        One-call workflow:
          1) scrape links
          2) optionally download files
          3) optionally extract preview text for each downloaded file
        Returns a dict with keys:
          - "links": List[str]
          - "downloads": List[metadata]
          - "previews": List[metadata] (if extract_preview True)
        """
        links = await self.scrape_pdf_links(max_pages=max_pages)
        result: Dict[str, object] = {"links": links, "downloads": [], "previews": []}

        if not download:
            return result

        downloads = await self.download_pdfs(links, overwrite=overwrite)
        result["downloads"] = downloads

        if extract_preview:
            previews = []
            # only for successfully downloaded files
            ok_paths = [d["path"] for d in downloads if d.get("status") == "ok"]
            for path in ok_paths:
                preview_text = await self.extract_pdf_preview(path)
                previews.append({"path": path, "preview": preview_text})
            result["previews"] = previews

        return result

    # -------------------------
    # Cleanup helpers
    # -------------------------
    async def close(self):
        """Close internal aiohttp session if any (call on shutdown)."""
        if self._session and not self._session.closed:
            await self._session.close()

    def cleanup_output_dir(self, keep_latest: Optional[int] = None):
        """
        Synchronous helper to remove files in the output directory.
        If keep_latest is specified, will keep the 'keep_latest' newest files.
        """
        files = sorted(self.output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if keep_latest is None:
            # remove all
            for f in files:
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                except Exception:
                    logger.exception("Error removing %s", f)
            return

        for f in files[keep_latest:]:
            try:
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()
            except Exception:
                logger.exception("Error removing %s", f)

