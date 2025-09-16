# import asyncio
# import logging
# import os
# import re
# import hashlib
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, urlparse
# from playwright.async_api import async_playwright
# import requests
# import fitz  # PyMuPDF
# from typing import List, Tuple, Optional


# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# # ---- Helpers for filenames ----
# def sanitize_filename(filename: str) -> str:
#     filename = filename.split("?")[0].split("#")[0]
#     return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


# def safe_filename(url: str, title: str = "", context: str = "", max_len: int = 80) -> str:
#     """Short, safe filename for a PDF using a slug + 12-char hash."""
#     h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
#     slug_raw = (title or context or os.path.basename(url.split("?")[0]) or "doc").strip()
#     slug_sanitized = sanitize_filename(slug_raw).strip("._-")
#     slug = slug_sanitized[:max_len] if slug_sanitized else "doc"
#     base = f"{slug}_{h}.pdf"
#     return base if len(base) <= (max_len + 20) else f"{h}.pdf"


# # ---- DB sources loader (optional) ----
# try:
#     from app.models import ScrapeSource  # type: ignore
# except Exception:  # pragma: no cover
#     ScrapeSource = None  # type: ignore


# def load_ministry_data_from_db() -> List[Tuple[str, str]]:
#     """Return (name, url) pairs from ScrapeSource, ordered by name. Empty on failure."""
#     entries: List[Tuple[str, str]] = []
#     if ScrapeSource is None:
#         return entries
#     try:
#         for s in ScrapeSource.objects.all().order_by("name"):
#             name = getattr(s, "name", "") or getattr(s, "ministry", "") or ""
#             url = getattr(s, "url", "")
#             if name and url:
#                 entries.append((name, url))
#     except Exception:
#         pass
#     return entries

# class PDFScraper:
#     def __init__(self, base_url, max_sections=20, headless=True):
#         self.base_url = base_url
#         self.max_sections = max_sections
#         self.headless = headless
#         self.pdf_links = set()

#     async def _extract_pdfs_from_html(self, html, base_url):
#         soup = BeautifulSoup(html, "html.parser")
#         new_links = {
#             urljoin(base_url, a["href"])
#             for a in soup.find_all("a", href=True)
#             if a["href"].lower().endswith(".pdf")
#         }
#         return new_links

#     async def _get_section_urls(self, page):
#         """Extract all section URLs from the main page."""
#         section_urls = set()
#         links = await page.query_selector_all('a[href*="?id="]')

#         for link in links:
#             href = await link.get_attribute("href")
#             if href:
#                 section_urls.add(urljoin(self.base_url, href))

#         logging.info(f"Found {len(section_urls)} section URLs")
#         return list(section_urls)

#     async def _process_page(self, page, url):
#         logging.info(f"Processing: {url}")
#         try:
#             await page.goto(url, wait_until="networkidle", timeout=60000)
#             await asyncio.sleep(3)

#             # Main content PDFs
#             html = await page.content()
#             base_domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
#             new_links = await self._extract_pdfs_from_html(html, base_domain)
#             self.pdf_links.update(new_links)
#             logging.info(f"Found {len(new_links)} PDFs on page")

#             # Frame PDFs
#             for i, frame in enumerate(page.frames[1:], 1):
#                 try:
#                     frame_html = await frame.content()
#                     frame_links = await self._extract_pdfs_from_html(frame_html, frame.url)
#                     new_count = len(frame_links - self.pdf_links)
#                     self.pdf_links.update(frame_links)
#                     if new_count > 0:
#                         logging.info(f"Frame {i}: Found {new_count} new PDFs")
#                 except Exception as e:
#                     logging.warning(f"Frame {i} error: {e}")

#             return True
#         except Exception as e:
#             logging.error(f"Failed to process page: {e}")
#             return False

#     async def run(self):
#         """Run the PDF scraper and return found links."""
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(
#                 headless=self.headless,
#                 channel="chrome",
#                 args=["--disable-blink-features=AutomationControlled"]
#             )

#             context = await browser.new_context(
#                 user_agent=(
#                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                     "AppleWebKit/537.36 (KHTML, like Gecko) "
#                     "Chrome/115.0.0.0 Safari/537.36"
#                 ),
#                 viewport={"width": 1920, "height": 1080}
#             )

#             page = await context.new_page()
#             await page.goto(self.base_url, wait_until="networkidle")
#             await asyncio.sleep(3)

#             section_urls = await self._get_section_urls(page)
#             all_urls = [self.base_url] + section_urls

#             for i, url in enumerate(all_urls[:self.max_sections], start=1):
#                 logging.info(f"Page {i}/{len(all_urls)}")
#                 success = await self._process_page(page, url)
#                 if not success:
#                     await page.reload(wait_until="networkidle")
#                     await self._process_page(page, url)

#             await browser.close()

#         return sorted(self.pdf_links)


# class PDFDownloader:
#     def __init__(self, output_dir: str, timeout: int = 10):
#         self.output_dir = output_dir
#         self.timeout = timeout
#         self.headers = {
#             "User-Agent": (
#                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                 "AppleWebKit/537.36 (KHTML, like Gecko) "
#                 "Chrome/115 Safari/537.36"
#             )
#         }
#         os.makedirs(self.output_dir, exist_ok=True)

#     def _download_pdf(self, url: str) -> Optional[str]:
#         """Download PDF from URL and return file path, or None if failed."""
#         filename = safe_filename(url)
#         path = os.path.join(self.output_dir, filename)

#         try:
#             r = requests.get(url, headers=self.headers, timeout=self.timeout)
#             content_type = r.headers.get("Content-Type", "").lower()

#             if r.status_code == 200 and ("application/pdf" in content_type or url.lower().endswith('.pdf')):
#                 with open(path, "wb") as f:
#                     f.write(r.content)
#                 logging.info(f"Downloaded: {filename}")
#                 return path
#             else:
#                 logging.warning(f"Invalid content type for {filename} ({content_type})")
#                 return None
#         except Exception as e:
#             logging.error(f"Error downloading {filename}: {e}")
#             return None

#     def _extract_text(self, pdf_path: str, max_chars: int = 1000) -> str:
#         """Extract text from PDF (first max_chars characters)."""
#         try:
#             doc = fitz.open(pdf_path)
#             text = "".join(page.get_text() for page in doc)
#             return text[:max_chars]
#         except Exception:
#             logging.warning(f"Corrupted or unreadable PDF: {os.path.basename(pdf_path)} (deleted)")
#             os.remove(pdf_path)
#             return ""

#     def run(self, pdf_links: List[str]) -> List[Tuple[str, str]]:
#         """
#         Download PDFs and extract snippets.
#         Returns a list of (filename, snippet) tuples.
#         """
#         pdf_texts = []
#         for url in pdf_links:
#             path = self._download_pdf(url)
#             if path:
#                 snippet = self._extract_text(path)
#                 if snippet:
#                     pdf_texts.append((os.path.basename(path), snippet))
#         logging.info(f"Completed. {len(pdf_texts)} valid PDFs saved in {self.output_dir}")
#         return pdf_texts
import asyncio
import logging
import os
import re
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Optional, Set

import requests
import fitz  # PyMuPDF
from typing import Any
try:
    from playwright.async_api import async_playwright, Page  # type: ignore
except Exception:
    async_playwright = None  # type: ignore
    Page = Any  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---- Helpers for filenames ----
def sanitize_filename(filename: str) -> str:
    """Remove problematic characters from a filename."""
    filename = filename.split("?")[0].split("#")[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def safe_filename(url: str, title: str = "", context: str = "", max_len: int = 80) -> str:
    """
    Create a short, safe filename for a PDF using a slug plus a 12‑character hash.

    The slug is derived from the URL, page title, or a provided context. A
    hash ensures uniqueness when multiple documents share the same name.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    slug_raw = (title or context or os.path.basename(url.split("?")[0]) or "doc").strip()
    slug_sanitized = sanitize_filename(slug_raw).strip("._-")
    slug = slug_sanitized[:max_len] if slug_sanitized else "doc"
    base = f"{slug}_{h}.pdf"
    # Avoid extremely long names
    return base if len(base) <= (max_len + 20) else f"{h}.pdf"


# ---- DB sources loader (optional) ----
try:
    from app.models import ScrapeSource  # type: ignore
except Exception:  # pragma: no cover
    ScrapeSource = None  # type: ignore


def load_ministry_data_from_db() -> List[Tuple[str, str]]:
    """
    Return a list of (name, url) pairs from ScrapeSource, ordered by name.

    If the Django model is unavailable, return an empty list. This helper
    allows the scraper to programmatically pull ministry URLs from the
    application's database.
    """
    entries: List[Tuple[str, str]] = []
    if ScrapeSource is None:
        return entries
    try:
        for s in ScrapeSource.objects.all().order_by("name"):
            name = getattr(s, "name", "") or getattr(s, "ministry", "") or ""
            url = getattr(s, "url", "")
            if name and url:
                entries.append((name, url))
    except Exception:
        pass
    return entries


class PDFScraper:
    """
    Crawl a website and gather links to PDF files.

    Parameters
    ----------
    base_url : str
        The starting page for the crawler. Only pages on this domain are explored.
    max_sections : int, optional
        Maximum number of pages to crawl (including the base page). Prevents
        infinite recursion on sites with many links. Default is 20.
    headless : bool, optional
        If True, the browser runs headless. Set to False for debugging. Default
        is True.
    """

    def __init__(self, base_url: str, max_sections: int = 20, headless: bool = True) -> None:
        self.base_url = base_url
        self.max_sections = max_sections
        self.headless = headless
        self.pdf_links: Set[str] = set()
        self.visited_urls: Set[str] = set()

    async def _extract_pdfs_from_html(self, html: str, base_url: str) -> Set[str]:
        """
        Parse an HTML string and extract all links ending with `.pdf`.

        Parameters
        ----------
        html : str
            The HTML content to search.
        base_url : str
            The base URL used to resolve relative links.
        Returns
        -------
        Set[str]
            A set of absolute PDF URLs.
        """
        soup = BeautifulSoup(html, "html.parser")
        new_links = {
            urljoin(base_url, a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        }
        return new_links

    async def _get_section_urls(self, page: "Page", current_url: str) -> List[str]:
        """
        Extract internal (same domain) links from a loaded page.

        Unlike the previous implementation that filtered on a specific query
        parameter, this method collects all anchor `href` values and
        normalizes them to absolute URLs. It then filters those URLs to
        include only those that:

        * Use the same scheme and netloc (domain) as the base URL.
        * Do not point to a PDF document (handled separately).
        * Have not already been visited.

        The intention is to crawl deeper into the site, discovering pages
        like "lois-et-reglementations/textes-juridiques" or "normes" on
        environnement.gov.ma/fr. By returning a list of new URLs,
        the caller can queue them for processing.

        Parameters
        ----------
        page : Page
            The Playwright page object currently loaded.
        current_url : str
            The URL of the page currently loaded.
        Returns
        -------
        List[str]
            A list of new internal URLs to crawl.
        """
        section_urls: Set[str] = set()
        # Determine the base domain to restrict to the same host
        parsed_base = urlparse(self.base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

        # Query all anchor tags with an href
        links = await page.query_selector_all("a[href]")
        for link in links:
            href = await link.get_attribute("href")
            if not href:
                continue
            href = href.strip()
            # Skip if anchor or JavaScript
            if href.startswith("#") or href.lower().startswith("javascript"):
                continue
            # Convert to absolute URL relative to the current page
            abs_url = urljoin(current_url, href)
            parsed = urlparse(abs_url)
            # Only follow links on the same domain
            if parsed.scheme and parsed.netloc == parsed_base.netloc:
                # Skip PDF files (already handled)
                if parsed.path.lower().endswith(".pdf"):
                    continue
                # Remove URL fragments and query parameters for normalization
                normalized = parsed._replace(fragment="", query="").geturl()
                # Avoid re‑visiting URLs
                if normalized not in self.visited_urls:
                    section_urls.add(normalized)

        logging.info(f"Found {len(section_urls)} new internal links on this page")
        return list(section_urls)

    async def _process_page(self, page: Page, url: str) -> bool:
        """
        Load a page, extract PDF links, and return success status.

        This method navigates to the provided URL, waits for network
        activity to settle, and then uses ``_extract_pdfs_from_html`` to
        identify PDF links on the page. It also inspects any frames on
        the page for additional PDF links.

        Parameters
        ----------
        page : Page
            The Playwright page object to use for navigation.
        url : str
            The URL to navigate to.
        Returns
        -------
        bool
            True if the page was processed successfully, False otherwise.
        """
        logging.info(f"Processing: {url}")
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            # Brief pause to allow dynamic content to load
            await asyncio.sleep(2)

            # Extract PDFs from the main document
            html = await page.content()
            base_domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            new_links = await self._extract_pdfs_from_html(html, base_domain)
            new_count = len(new_links - self.pdf_links)
            self.pdf_links.update(new_links)
            if new_count > 0:
                logging.info(f"Main page: found {new_count} new PDFs")

            # Extract PDFs from each frame (iframe or frame)
            for i, frame in enumerate(page.frames[1:], 1):
                try:
                    frame_html = await frame.content()
                    frame_links = await self._extract_pdfs_from_html(frame_html, frame.url)
                    added = len(frame_links - self.pdf_links)
                    self.pdf_links.update(frame_links)
                    if added > 0:
                        logging.info(f"Frame {i}: found {added} new PDFs")
                except Exception as e:
                    logging.warning(f"Frame {i} error: {e}")

            return True
        except Exception as e:
            logging.error(f"Failed to process page {url}: {e}")
            return False

    async def run(self) -> List[str]:
        """
        Crawl the website starting from the base URL and return sorted PDF links.

        The crawler performs a breadth‑first search (BFS) across internal pages.
        It maintains a queue of URLs to visit (``to_visit``) and stops when
        either ``max_sections`` pages have been processed or no more new
        URLs remain. PDF links found during the crawl are stored in
        ``self.pdf_links``.

        Returns
        -------
        List[str]
            A sorted list of unique PDF URLs.
        """
        if async_playwright is None:
            raise ImportError(
                "playwright is not installed. Install it with `pip install playwright` "
                "and run `playwright install chromium` before using PDFScraper.run()"
            )
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = await context.new_page()

            # BFS queue for pages to visit
            to_visit: List[str] = []
            # Normalize the base URL (remove trailing slash for consistency)
            parsed_base = urlparse(self.base_url)
            base_norm = parsed_base._replace(fragment="", query="").geturl().rstrip("/")
            to_visit.append(base_norm)

            while to_visit and len(self.visited_urls) < self.max_sections:
                current_url = to_visit.pop(0)
                if current_url in self.visited_urls:
                    continue
                self.visited_urls.add(current_url)
                success = await self._process_page(page, current_url)
                # If successful, gather new internal links from this page
                if success:
                    new_sections = await self._get_section_urls(page, current_url)
                    # Append new links to the queue
                    for link in new_sections:
                        if link not in self.visited_urls and link not in to_visit:
                            to_visit.append(link)

            await browser.close()
        return sorted(self.pdf_links)


class PDFDownloader:
    """
    Download a list of PDF files and extract text snippets.

    Parameters
    ----------
    output_dir : str
        Directory where downloaded PDFs will be saved.
    timeout : int, optional
        Connection timeout in seconds for each download. Default is 10.
    """

    def __init__(self, output_dir: str, timeout: int = 10) -> None:
        self.output_dir = output_dir
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115 Safari/537.36"
            )
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def _download_pdf(self, url: str) -> Optional[str]:
        """
        Download a PDF from a URL and return its local file path.

        If the HTTP request fails, or if the content type is not PDF,
        the method returns None. Filenames are generated using
        ``safe_filename``. This helper is used internally by ``run``.
        """
        filename = safe_filename(url)
        path = os.path.join(self.output_dir, filename)
        try:
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            content_type = r.headers.get("Content-Type", "").lower()
            if r.status_code == 200 and ("application/pdf" in content_type or url.lower().endswith(".pdf")):
                with open(path, "wb") as f:
                    f.write(r.content)
                logging.info(f"Downloaded: {filename}")
                return path
            else:
                logging.warning(f"Invalid content type for {filename} ({content_type})")
                return None
        except Exception as e:
            logging.error(f"Error downloading {filename}: {e}")
            return None

    def _extract_text(self, pdf_path: str, max_chars: int = 1000) -> str:
        """
        Extract text from the first ``max_chars`` characters of a PDF.

        If the PDF is corrupt or cannot be parsed, the file is removed
        and an empty string is returned.
        """
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            return text[:max_chars]
        except Exception:
            logging.warning(f"Corrupted or unreadable PDF: {os.path.basename(pdf_path)} (deleted)")
            os.remove(pdf_path)
            return ""

    def run(self, pdf_links: List[str]) -> List[Tuple[str, str]]:
        """
        Download a list of PDF links and extract text snippets from each.

        Parameters
        ----------
        pdf_links : List[str]
            A list of PDF URLs to download and process.
        Returns
        -------
        List[Tuple[str, str]]
            A list of tuples containing the filename and a text snippet.
        """
        pdf_texts: List[Tuple[str, str]] = []
        for url in pdf_links:
            path = self._download_pdf(url)
            if path:
                snippet = self._extract_text(path)
                if snippet:
                    pdf_texts.append((os.path.basename(path), snippet))
        logging.info(f"Completed. {len(pdf_texts)} valid PDFs saved in {self.output_dir}")
        return pdf_texts


async def _main_async(url: str, max_sections: int = 20, out_dir: str = "./downloads") -> None:
    """Helper for running the scraper and downloader as a standalone script."""
    scraper = PDFScraper(url, max_sections=max_sections, headless=True)
    pdf_links = await scraper.run()
    logging.info(f"Found {len(pdf_links)} PDF links:")
    for link in pdf_links:
        logging.info(link)
    downloader = PDFDownloader(out_dir)
    results = downloader.run(pdf_links)
    logging.info("Extraction complete. Sample snippets:")
    for fname, snippet in results[:5]:
        logging.info(f"{fname}: {snippet[:80]}...")

