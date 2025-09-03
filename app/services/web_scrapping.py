import asyncio
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
import os
import logging
import requests
import fitz  # PyMuPDF
from typing import List, Tuple


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class PDFScraper:
    def __init__(self, base_url, max_sections=20, headless=True):
        self.base_url = base_url
        self.max_sections = max_sections
        self.headless = headless
        self.pdf_links = set()

    async def _extract_pdfs_from_html(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        new_links = {
            urljoin(base_url, a["href"])
            for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        }
        return new_links

    async def _get_section_urls(self, page):
        """Extract all section URLs from the main page."""
        section_urls = set()
        links = await page.query_selector_all('a[href*="?id="]')

        for link in links:
            href = await link.get_attribute("href")
            if href:
                section_urls.add(urljoin(self.base_url, href))

        logging.info(f"Found {len(section_urls)} section URLs")
        return list(section_urls)

    async def _process_page(self, page, url):
        logging.info(f"Processing: {url}")
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(3)

            # Main content PDFs
            html = await page.content()
            base_domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            new_links = await self._extract_pdfs_from_html(html, base_domain)
            self.pdf_links.update(new_links)
            logging.info(f"Found {len(new_links)} PDFs on page")

            # Frame PDFs
            for i, frame in enumerate(page.frames[1:], 1):
                try:
                    frame_html = await frame.content()
                    frame_links = await self._extract_pdfs_from_html(frame_html, frame.url)
                    new_count = len(frame_links - self.pdf_links)
                    self.pdf_links.update(frame_links)
                    if new_count > 0:
                        logging.info(f"Frame {i}: Found {new_count} new PDFs")
                except Exception as e:
                    logging.warning(f"Frame {i} error: {e}")

            return True
        except Exception as e:
            logging.error(f"Failed to process page: {e}")
            return False

    async def run(self):
        """Run the PDF scraper and return found links."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"]
            )

            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080}
            )

            page = await context.new_page()
            await page.goto(self.base_url, wait_until="networkidle")
            await asyncio.sleep(3)

            section_urls = await self._get_section_urls(page)
            all_urls = [self.base_url] + section_urls

            for i, url in enumerate(all_urls[:self.max_sections], start=1):
                logging.info(f"Page {i}/{len(all_urls)}")
                success = await self._process_page(page, url)
                if not success:
                    await page.reload(wait_until="networkidle")
                    await self._process_page(page, url)

            await browser.close()

        return sorted(self.pdf_links)


class PDFDownloader:
    def __init__(self, output_dir: str, timeout: int = 10):
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

    def _download_pdf(self, url: str) -> str:
        """Download PDF from URL and return file path, or None if failed."""
        filename = os.path.basename(url.split("?")[0])  # Strip query params
        path = os.path.join(self.output_dir, filename)

        try:
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            content_type = r.headers.get("Content-Type", "").lower()

            if r.status_code == 200 and "application/pdf" in content_type:
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
        """Extract text from PDF (first max_chars characters)."""
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
        Download PDFs and extract snippets.
        Returns a list of (filename, snippet) tuples.
        """
        pdf_texts = []
        for url in pdf_links:
            path = self._download_pdf(url)
            if path:
                snippet = self._extract_text(path)
                if snippet:
                    pdf_texts.append((os.path.basename(path), snippet))
        logging.info(f"Completed. {len(pdf_texts)} valid PDFs saved in {self.output_dir}")
        return pdf_texts