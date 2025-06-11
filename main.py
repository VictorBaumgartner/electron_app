import shutil # Add this import at the top
import asyncio
import os
import json
import re
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from typing import List, Dict, Any, Tuple, Optional, Set # Added Set
import csv
import io
from concurrent.futures import ThreadPoolExecutor
import aiohttp # Add missing aiohttp import
import psutil # For system monitoring
import platform # For machine name
from fastapi import FastAPI, APIRouter, HTTPException # Added FastAPI imports
from pydantic import BaseModel # Added BaseModel
# --- Sitemap Processing Imports ---
import logging
import sys
from pathlib import Path # Added for Path object
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app and router
app = FastAPI()
router = APIRouter()

# --- End Logging Setup ---

# Force UTF-8 encoding at the top of the script
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- Constants ---
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# Define the exclusion keywords for filenames (case-insensitive check will be used)
EXCLUDE_KEYWORDS = ['pdf', 'jpeg', 'jpg', 'png', 'webp']

# Global state to track if a crawl is in progress
_is_processing = False

# --- Pydantic Models for API Requests ---
class FetchUrlsRequest(BaseModel):
    master_server_ip: str
    master_server_port: int

class SendStatusRequest(BaseModel):
    master_server_ip: str
    master_server_port: int

class StartCrawlRequest(BaseModel):
    urls: List[str]
    output_dir: str
    max_concurrency: int = 8
    max_depth: int = 2

class NotifyFinishedRequest(BaseModel):
    master_server_ip: str
    master_server_port: int
    processed_successfully: bool = True

# --- Import the function from sitemap_crawler.py ---
try:
    from sitemap_crawler import get_sitemap_data_for_single_url
    SITEMAP_CRAWLER_AVAILABLE = True
    logger.info("Successfully imported sitemap_crawler module.")
except ImportError:
    SITEMAP_CRAWLER_AVAILABLE = False
    logger.warning("sitemap_crawler.py not found or 'get_sitemap_data_for_single_url' not available. Sitemap processing will be basic or skipped.")
    async def get_sitemap_data_for_single_url(effective_start_url: str, session: aiohttp.ClientSession, *args, **kwargs) -> List[Tuple[str, str]]: # Dummy function
        logger.error("Sitemap crawler module was not imported correctly. Cannot get sitemap data.")
        return []

# --- URL Utilities ---
def prepare_initial_url_scheme(url_str: str) -> str:
    """Ensures the URL has a scheme, defaulting to http if none."""
    if not url_str: return ""
    url_str = url_str.strip()
    parsed = urlparse(url_str)
    if not parsed.scheme:
        temp_path_part = parsed.path.split('/')[0] if parsed.path else ""
        if not parsed.netloc and temp_path_part and ('.' in temp_path_part or temp_path_part == 'localhost'):
             return f"http://{url_str.lstrip('//')}"
        elif parsed.netloc and not parsed.scheme:
             return f"http:{url_str}" 
        else: 
            return f"http://{url_str.lstrip('//')}"
    if parsed.scheme.lower() not in ['http', 'https']:
        logger.warning(f"URL {url_str} has an unusual scheme '{parsed.scheme}'. Crawler primarily supports http/https.")
    return url_str


@router.get("/metrics") # Replaced eel.expose
def get_local_machine_metrics():
    """
    Returns local machine metrics (name, storage, capacity) and current crawling status.
    """
    try:
        # Machine Name
        machine_name = platform.node()
        # Total Storage
        total_disk_gb = 0
        free_disk_gb = 0
        for part in psutil.disk_partitions(all=False):
            if os.name == 'nt': # Windows
                if 'cdrom' in part.opts or part.fstype == '':
                    continue
            usage = psutil.disk_usage(part.mountpoint)
            total_disk_gb += usage.total / (1024**3)
            free_disk_gb += usage.free / (1024**3)
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1) # Blocking call, non-blocking for subsequent calls

        # Memory Usage
        virtual_memory = psutil.virtual_memory()
        memory_percent = virtual_memory.percent

        return {
            "status": "success",
            "machine_name": machine_name,
            "total_storage_gb": round(total_disk_gb, 2),
            "free_storage_gb": round(free_disk_gb, 2),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory_percent,
            "crawling_status": "in_use" if _is_processing else "idle"
        }
    except Exception as e:
        logger.error(f"Error getting local machine metrics: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to retrieve machine metrics: {str(e)}"}

@router.post("/fetch_urls") # Replaced eel.expose
async def fetch_urls_from_master_server(request: FetchUrlsRequest) -> Dict[str, Any]:
    """
    Fetches a list of URLs to crawl from the central master server's API endpoint.
    """
    master_server_ip = request.master_server_ip
    master_server_port = request.master_server_port
    target_url = f"http://{master_server_ip}:{master_server_port}/get_urls_to_crawl"
    logger.info(f"Attempting to fetch URLs from master server: {target_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    urls = data.get("urls", [])
                    message = data.get("message", f"Successfully fetched {len(urls)} URLs.")
                    logger.info(f"Fetched URLs from {master_server_ip}:{master_server_port}: {message}")
                    return {"status": "success", "urls": urls, "message": message}
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch URLs from {master_server_ip}:{master_server_port}. HTTP Status: {response.status}, Response: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Failed to fetch URLs: Server responded with HTTP {response.status}. {error_text}")
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Could not connect to master server {master_server_ip}:{master_server_port}: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to the master server ({master_server_ip}:{master_server_port}). Please ensure it is running and accessible. Error: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout when trying to connect to master server {master_server_ip}:{master_server_port}")
        raise HTTPException(status_code=504, detail=f"Connection to master server ({master_server_ip}:{master_server_port}) timed out.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching URLs from master server {master_server_ip}:{master_server_port}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching URLs: {str(e)}")

@router.post("/send_status") # Replaced eel.expose
async def send_local_machine_status_to_master(request: SendStatusRequest):
    """
    Sends the local machine's current metrics and crawling status to the master server.
    """
    master_server_ip = request.master_server_ip
    master_server_port = request.master_server_port
    metrics = get_local_machine_metrics() # This function returns a dict
    if metrics["status"] == "error":
        logger.error(f"Failed to get local machine metrics for status update: {metrics["message"]}")
        raise HTTPException(status_code=500, detail="Failed to get local machine metrics for status update.")

    target_url = f"http://{master_server_ip}:{master_server_port}/update_machine_status"
    payload = {
        "machine_name": metrics["machine_name"],
        "total_storage_gb": metrics["total_storage_gb"],
        "free_storage_gb": metrics["free_storage_gb"],
        "cpu_usage_percent": metrics["cpu_usage_percent"],
        "memory_usage_percent": metrics["memory_usage_percent"],
        "crawling_status": metrics["crawling_status"]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(target_url, json=payload, timeout=5) as response:
                response_data = await response.json()
                if response.status == 200:
                    logger.debug(f"Successfully sent status to master: {response_data.get('message')}")
                    return {"status": "success", "message": response_data.get('message', 'Status sent.')}
                else:
                    error_text = response_data.get('detail', await response.text())
                    logger.warning(f"Master server responded with {response.status} when receiving status: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Master server responded with HTTP {response.status}: {error_text}")
    except aiohttp.ClientConnectorError as e:
        logger.warning(f"Could not connect to master server {master_server_ip}:{master_server_port} for status update: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to master server for status update: {e}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when sending status to master server {master_server_ip}:{master_server_port}.")
        raise HTTPException(status_code=504, detail=f"Timeout when sending status to master server.")
    except Exception as e:
        logger.error(f"Unexpected error sending status to master server {master_server_ip}:{master_server_port}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error sending status: {str(e)}")

@router.get("/machines_status") # Replaced eel.expose
async def get_all_machines_status_from_master(master_server_ip: str, master_server_port: int) -> Dict[str, Any]:
    """
    Fetches the status of all connected client machines from the master server.
    """
    target_url = f"http://{master_server_ip}:{master_server_port}/get_all_machines_status"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Successfully fetched all machines status from master.")
                    return {"status": "success", "machines": data.get("machines", [])}
                else:
                    error_text = await response.text()
                    logger.warning(f"Master server responded with {response.status} when getting all machines status: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Failed to fetch all machine statuses: HTTP {response.status}. {error_text}")
    except aiohttp.ClientConnectorError as e:
        logger.warning(f"Could not connect to master server {master_server_ip}:{master_server_port} to get all machines status: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to master server to get all machine statuses: {e}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout when getting all machines status from master server {master_server_ip}:{master_server_port}.")
        raise HTTPException(status_code=504, detail=f"Timeout when getting all machine statuses from master server.")
    except Exception as e:
        logger.error(f"Unexpected error fetching all machines status from master server {master_server_ip}:{master_server_port}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching all machine statuses: {str(e)}")

# --- Helper function to create a ZIP archive ---
def create_zip_archive(source_dir_path: str, output_zip_base_name: str) -> Optional[str]:
    """
    Creates a ZIP archive from the source_dir_path.
    The output_zip_base_name should be the desired path for the zip file *without* the .zip extension.
    e.g., output_zip_base_name = /path/to/archive_name
    Returns the full path to the created ZIP file, or None on error.
    """
    try:
        # shutil.make_archive will append ".zip" to output_zip_base_name
        # root_dir is the directory that will be the root of the archive.
        # base_dir is the directory to start archiving from, relative to root_dir.
        # If base_dir is the same as the last component of source_dir_path, and
        # root_dir is its parent, then the archive will contain the source_dir_path itself.

        source_dir_abs = os.path.abspath(source_dir_path)
        root_dir = os.path.dirname(source_dir_abs)
        base_dir_to_zip = os.path.basename(source_dir_abs)

        logger.info(f"Attempting to create ZIP: output_base='{output_zip_base_name}', root_dir='{root_dir}', base_dir_to_zip='{base_dir_to_zip}'")

        zip_file_path = shutil.make_archive(
            base_name=output_zip_base_name,
            format='zip',
            root_dir=root_dir,
            base_dir=base_dir_to_zip
        )
        logger.info(f"Successfully created ZIP archive: {zip_file_path}")
        return zip_file_path
    except Exception as e:
        logger.error(f"Error creating ZIP archive for {source_dir_path}: {e}", exc_info=True)
        return None

def normalize_url_for_deduplication(url_string: str) -> str:
    """Removes the fragment from a URL and lowercases scheme/netloc for deduplication purposes."""
    try:
        parsed = urlparse(url_string)
        path = parsed.path
        if path and not path.startswith('/'):
            path = '/' + path
        elif not path and (parsed.query or parsed.params):
            path = '/'
        return urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path or '/', 
            parsed.params,
            parsed.query, 
            '' 
        )).rstrip('/')
    except Exception as e:
        logger.warning(f"Could not normalize URL '{url_string}' for deduplication: {e}. Returning original after basic processing.")
        return url_string.rstrip('/')

async def resolve_initial_url(session: aiohttp.ClientSession, url_to_resolve: str) -> Tuple[Optional[str], Optional[str]]:
    logger.debug(f"Attempting to resolve: {url_to_resolve} with headers: {session.headers}")
    try:
        async with session.get(url_to_resolve, allow_redirects=True, timeout=20) as response:
            effective_url = str(response.url)
            if response.status >= 400:
                error_msg = f"Initial URL resolution for '{url_to_resolve}' resulted in HTTP {response.status} at final URL '{effective_url}'"
                logger.warning(error_msg)
                if response.status == 403:
                    logger.debug(f"403 Response headers for {effective_url}: {response.headers}")
                return None, error_msg
            logger.info(f"Initial URL '{url_to_resolve}' resolved to effective URL '{effective_url}' (status: {response.status})")
            return effective_url, None
    except asyncio.TimeoutError:
        error_msg = f"Timeout resolving initial URL '{url_to_resolve}'"
        logger.warning(error_msg)
        return None, error_msg
    except aiohttp.ClientError as e:
        error_msg = f"ClientError resolving initial URL '{url_to_resolve}': {e}"
        logger.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error resolving initial URL '{url_to_resolve}': {e}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

def clean_markdown(md_text: str) -> str:
    md_text = re.sub(r'!\[([^]]*)\]\((http[s]?://[^)]+)\)', '', md_text)
    md_text = re.sub(r'\[([^]]+)\]\((http[s]?://[^)]+)\)', r'\1', md_text)
    md_text = re.sub(r'(?<!\]\)https?://\S+', '', md_text)
    md_text = re.sub(r'\[[^?]\d+\]', '', md_text)
    md_text = re.sub(r'^[^?]\d+:\s?.*$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^[ \t]+$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'(\*\*|_)(.*?)\1', r'\2', md_text)
    md_text = re.sub(r'^[ \t]*#+[ \t]*$', '', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'(\(\))', '', md_text)
    md_text = re.sub(r'\n[ \t]*\n+', '\n\n', md_text)
    md_text = re.sub(r'[ \t]+', ' ', md_text)
    return md_text.strip()

def read_urls_from_csv(csv_content: str) -> List[str]:
    urls = []
    csvfile = io.StringIO(csv_content)
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if not row:
            continue
        url_input = row[0].strip()
        if url_input:
            schemed_url = prepare_initial_url_scheme(url_input)
            try:
                parsed_url = urlparse(schemed_url)
                if parsed_url.netloc: 
                    urls.append(schemed_url)
                else:
                    logger.warning(f"Skipping URL with no recognizable domain after schematization on line {i+1}: '{url_input}' -> '{schemed_url}'")
            except Exception:
                logger.warning(f"Skipping invalid URL format on line {i+1}: '{url_input}'")
        else:
            logger.warning(f"Skipping empty entry on line {i+1}: '{row[0]}'")
    return urls

def sanitize_filename(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace(".", "_")
        path = parsed.path.strip("/").replace("/", "_").replace(".", "_")
        if not path:
            path = "index"
        query = parsed.query
        if query:
            query = query[:50] 
            query_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', query)
            filename_base = f"{netloc}_{path}_{query_safe}"
        else:
            filename_base = f"{netloc}_{path}"
        filename_base = re.sub(r'[<>:"/\\|?*]', '_', filename_base) 
        filename_base = re.sub(r'[ \._-]+', '_', filename_base) 
        filename_base = re.sub(r'^_*', '', filename_base) 
        filename_base = re.sub(r'_*$', '', filename_base) 
        if not filename_base: 
            filename_base = f"url_{abs(hash(url))}"
        max_len_without_suffix = 255 - len(".md") - 5 
        filename = filename_base[:max_len_without_suffix] + ".md"
        return filename
    except Exception as e:
        logger.error(f"Error sanitizing URL filename for {url}: {e}")
        return f"error_parsing_{abs(hash(url))}.md"

def sanitize_dirname(url: str) -> str:
    try:
        parsed = urlparse(url)
        dirname_base = parsed.netloc
        dirname_base = re.sub(r'[<>:"/\\|?*]', '_', dirname_base)
        dirname_base = re.sub(r'[ \._-]+', '_', dirname_base)
        dirname_base = re.sub(r'^_*', '', dirname_base)
        dirname_base = re.sub(r'_*$', '', dirname_base)
        if not dirname_base:
            dirname_base = f"domain_{abs(hash(url))}"
        return dirname_base[:150]
    except Exception as e:
        logger.error(f"Error sanitizing URL directory name for {url}: {e}")
        return f"domain_error_{abs(hash(url))}"

CrawlQueueItem = Tuple[str, int, str, str]

def process_markdown_and_save(url: str, markdown_content: str, output_path: str) -> Dict[str, Any]:
    try:
        cleaned_markdown = clean_markdown(markdown_content)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.access(os.path.dirname(output_path), os.W_OK):
            raise OSError(f"No write permission for directory: {os.path.dirname(output_path)}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Original URL (effective for crawl): {url}\n\n{cleaned_markdown}\n")
        if os.path.exists(output_path):
            logger.info(f"Saved cleaned Markdown to: {output_path}")
            return {"status": "success", "url": url, "path": output_path}
        else: 
            logger.error(f"File write appeared successful but file not found: {output_path}")
            raise IOError(f"File was not created or accessible: {output_path}")
    except Exception as e:
        logger.error(f"Error processing/saving Markdown for {url} to {output_path}: {e}", exc_info=True)
        return {"status": "failed", "url": url, "error": str(e)}

async def crawl_website_single_site(
    start_url_original_schemed: str, 
    output_dir: str,
    max_concurrency: int,
    max_depth: int
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "success": [], "failed": [], "skipped_by_filter": [],
        "initial_url": start_url_original_schemed, 
        "effective_start_url": None,
        "output_path_for_site": None
    }
    effective_start_url: Optional[str] = None
    effective_start_domain: Optional[str] = None
    site_output_path_specific: Optional[str] = None

    async with aiohttp.ClientSession(headers=COMMON_HEADERS) as http_session:
        resolved_url, error_msg = await resolve_initial_url(http_session, start_url_original_schemed)
        if error_msg or not resolved_url:
            results["failed"].append({"url": start_url_original_schemed, "error": f"Initial URL resolution failed: {error_msg or 'Unknown error'}"})
            logger.error(f"Aborting crawl for {start_url_original_schemed} due to resolution failure.")
            return results
        effective_start_url = resolved_url
        results["effective_start_url"] = effective_start_url

    try:
        parsed_effective_start_url = urlparse(effective_start_url)
        effective_start_domain = parsed_effective_start_url.netloc
        if not effective_start_domain:
            error_msg = f"Could not extract domain from effective start URL: {effective_start_url} (original: {start_url_original_schemed})"
            results["failed"].append({"url": start_url_original_schemed, "effective_url_failed": effective_start_url, "error": error_msg})
            logger.error(error_msg)
            return results
        site_subdir_name = sanitize_dirname(effective_start_url)
        site_output_path_specific = os.path.join(output_dir, site_subdir_name)
        site_output_path_specific = os.path.abspath(site_output_path_specific)
        results["output_path_for_site"] = site_output_path_specific
        logger.info(f"Effective crawl target domain: {effective_start_domain} (from {effective_start_url})")
        logger.info(f"Saving files for this site in: {site_output_path_specific}")
        os.makedirs(site_output_path_specific, exist_ok=True)
        if not os.path.isdir(site_output_path_specific): 
            raise OSError(f"Failed to create or access directory: {site_output_path_specific}")
    except Exception as e:
        error_msg = f"Error setting up directories for effective URL {effective_start_url} (original: {start_url_original_schemed}): {e}"
        results["failed"].append({"url": start_url_original_schemed, "error": error_msg})
        logger.error(f"Error setting up for crawl: {error_msg}", exc_info=True)
        return results

    crawled_urls: Set[str] = set()
    queued_urls: Set[str] = set()
    crawl_queue: asyncio.Queue[CrawlQueueItem] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrency)
    normalized_effective_start_url = normalize_url_for_deduplication(effective_start_url)
    crawl_queue.put_nowait((normalized_effective_start_url, 0, effective_start_domain, site_output_path_specific))
    queued_urls.add(normalized_effective_start_url)
    logger.info(f"Starting crawl for (original): {start_url_original_schemed}, effective start: {effective_start_url} with max_depth={max_depth}, max_concurrency={max_concurrency}")
    md_generator = DefaultMarkdownGenerator(options={"ignore_links": True, "escape_html": True, "body_width": 0})
    config = CrawlerRunConfig(markdown_generator=md_generator, cache_mode="BYPASS", exclude_social_media_links=True)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        async def crawl_page_worker():
            nonlocal site_output_path_specific
            while True:
                current_url_from_queue, current_depth, expected_domain, current_site_output_path = "", 0, "", ""
                try:
                    current_url_from_queue, current_depth, expected_domain, current_site_output_path = await asyncio.wait_for(crawl_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if crawl_queue.empty(): 
                        break
                    continue 
                except asyncio.CancelledError:
                    logger.info("Crawl page worker cancelled.")
                    break 
                if current_url_from_queue in crawled_urls:
                    crawl_queue.task_done()
                    continue
                crawled_urls.add(current_url_from_queue)
                logger.info(f"Crawling ({len(crawled_urls)}): {current_url_from_queue} (Depth: {current_depth})")
                filename = sanitize_filename(current_url_from_queue) 
                output_path = os.path.join(current_site_output_path, filename)
                path_for_exclude_check = urlparse(current_url_from_queue).path.lower()
                if any(keyword in filename.lower() or f".{keyword}" in path_for_exclude_check for keyword in EXCLUDE_KEYWORDS):
                    logger.info(f"Skipping save for {current_url_from_queue} due to filename/URL path filter: {filename}")
                    results["skipped_by_filter"].append(current_url_from_queue)
                    if current_depth < max_depth:
                        page_data_for_links = None
                        try:
                            async with semaphore: 
                                async with AsyncWebCrawler(verbose=False) as crawler:
                                    page_data_for_links = await crawler.arun(url=current_url_from_queue, config=config)
                            if page_data_for_links and page_data_for_links.success and page_data_for_links.links:
                                final_page_url_for_links = normalize_url_for_deduplication(page_data_for_links.url or current_url_from_queue)
                                for link_info in page_data_for_links.links.get("internal", []): 
                                    href = link_info.get("href")
                                    if not href: continue
                                    try:
                                        abs_link = urljoin(final_page_url_for_links, href)
                                        norm_abs_link = normalize_url_for_deduplication(abs_link)
                                        parsed_norm_abs_link = urlparse(norm_abs_link)
                                        if parsed_norm_abs_link.scheme in ('http', 'https') and parsed_norm_abs_link.netloc == expected_domain and norm_abs_link not in crawled_urls and norm_abs_link not in queued_urls:
                                            await crawl_queue.put((norm_abs_link, current_depth + 1, expected_domain, current_site_output_path))
                                            queued_urls.add(norm_abs_link)
                                    except Exception as link_e: logger.warning(f"Link processing error for '{href}' (from {final_page_url_for_links}): {link_e}")
                            elif page_data_for_links and not page_data_for_links.success:
                                logger.warning(f"Link extraction failed (skipped save) for {current_url_from_queue}: {page_data_for_links.error_message}")
                        except Exception as crawl_err:
                            logger.error(f"Error during link extraction (skipped save) for {current_url_from_queue}: {crawl_err}", exc_info=False)
                    crawl_queue.task_done()
                    continue
                page_data = None
                try:
                    async with semaphore: 
                        async with AsyncWebCrawler(verbose=False) as crawler:
                            page_data = await crawler.arun(url=current_url_from_queue, config=config)
                    final_url_processed = normalize_url_for_deduplication(page_data.url or current_url_from_queue)
                    if final_url_processed != current_url_from_queue and final_url_processed in crawled_urls:
                        logger.info(f"{current_url_from_queue} redirected to already crawled {final_url_processed}. Skipping save/link processing.")
                        crawl_queue.task_done()
                        continue
                    if final_url_processed != current_url_from_queue:
                        crawled_urls.add(final_url_processed)
                    if page_data.success and page_data.markdown:
                        loop = asyncio.get_running_loop()
                        process_result = await loop.run_in_executor(
                            executor, process_markdown_and_save, final_url_processed, page_data.markdown.raw_markdown, output_path
                        )
                        if process_result["status"] == "success":
                            results["success"].append(final_url_processed)
                        else:
                            results["failed"].append({"url": final_url_processed, "error": process_result["error"]})
                    elif not page_data.success:
                        logger.warning(f"Failed to crawl {current_url_from_queue} (final: {final_url_processed}): {page_data.error_message}")
                        results["failed"].append({"url": current_url_from_queue, "final_url_if_redirected": final_url_processed, "error": page_data.error_message or "Unknown crawl error"})
                    elif not page_data.markdown: 
                        logger.warning(f"Crawled {final_url_processed} successfully but no Markdown content.")
                        results["failed"].append({"url": final_url_processed, "error": "No Markdown content generated"})
                    if page_data.success and current_depth < max_depth and page_data.links:
                        for link_info in page_data.links.get("internal", []): 
                            href = link_info.get("href")
                            if not href: continue
                            try:
                                absolute_url = urljoin(final_url_processed, href) 
                                normalized_absolute_url = normalize_url_for_deduplication(absolute_url)
                                parsed_normalized_absolute_url = urlparse(normalized_absolute_url)
                                if parsed_normalized_absolute_url.scheme in ('http', 'https') and parsed_normalized_absolute_url.netloc == expected_domain: 
                                    if normalized_absolute_url not in crawled_urls and normalized_absolute_url not in queued_urls:
                                        await crawl_queue.put((normalized_absolute_url, current_depth + 1, expected_domain, current_site_output_path))
                                        queued_urls.add(normalized_absolute_url)
                            except Exception as link_e:
                                logger.warning(f"Error processing link '{href}' (from {final_url_processed}): {link_e}")
                except Exception as e_crawl:
                    url_in_error = page_data.url if page_data and page_data.url else current_url_from_queue
                    logger.error(f"Error in crawl_page_worker for {url_in_error}: {e_crawl}", exc_info=True)
                    results["failed"].append({"url": url_in_error, "error": f"Worker exception: {str(e_crawl)}"})
                finally:
                    crawl_queue.task_done()
        worker_tasks = [asyncio.create_task(crawl_page_worker()) for _ in range(max_concurrency)]
        await crawl_queue.join()
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
    logger.info(f"Finished crawl processing for (original): {start_url_original_schemed}, effective: {effective_start_url}")
    return results

# --- Sitemap Processing Functions ---
async def process_and_save_sitemap(effective_site_url: str, site_specific_output_path: str) -> Dict[str, Any]:
    sitemap_results: Dict[str, Any] = {
        "status": "initiated", "sitemap_csv_path": None,
        "sitemap_urls_found_with_valid_lastmod": 0,
        "total_sitemap_entries_returned_by_crawler_module": 0, # New key
        "error": None,
        "based_on_effective_url": effective_site_url
    }
    sitemap_csv_full_path = Path(site_specific_output_path) / "sitemap_data.csv"
    
    if not effective_site_url: # Should not happen if called correctly
        sitemap_results.update({"status": "skipped_no_effective_url", "error": "Effective site URL was not provided."})
        return sitemap_results
        
    try:
        # Use COMMON_HEADERS for the session passed to the sitemap crawler module
        async with aiohttp.ClientSession(headers=COMMON_HEADERS) as session:
            # get_sitemap_data_for_single_url (from sitemap_crawler.py) returns:
            # List[Tuple[normalized_url_str, iso_lastmod_str]]
            # - URLs are already normalized by sitemap_crawler's normalize_url_for_sitemap_processing.
            # - Lastmod is already a valid ISO string (or the entry wouldn't be returned).
            # - Uniqueness of URLs is already handled by the Dict within get_sitemap_data_for_single_url.
            sitemap_entries_from_module = await get_sitemap_data_for_single_url(effective_site_url, session)
        
        # This count represents entries that the sitemap_crawler module found AND considered valid 
        # (i.e., had a parsable lastmod and were unique).
        sitemap_results["total_sitemap_entries_returned_by_crawler_module"] = len(sitemap_entries_from_module)

        if not sitemap_entries_from_module:
            # This means get_sitemap_data_for_single_url returned an empty list.
            # The sitemap_crawler.py's own logs (e.g., "Sitemap: Finalized 0 unique entries...") 
            # should provide details if debugging is needed for a specific site.
            error_msg = "The sitemap crawler module did not find any sitemap entries with valid/parsable lastmod timestamps."
            if not SITEMAP_CRAWLER_AVAILABLE: # Check if the dummy function was used
                 error_msg = "Sitemap crawler module is not available. " + error_msg.lower() # make it flow better
           
            sitemap_results.update({"status": "no_valid_sitemap_data_found_by_module", "error": error_msg})
            logger.info(f"Sitemap: {error_msg} For {effective_site_url}.")
            return sitemap_results

        # sitemap_entries_from_module is already the List[Tuple[str, str]] we need for the CSV.
        processed_for_csv = sitemap_entries_from_module

        # The original error message ("Sitemap entries fetched, but none had valid/parsable lastmod timestamps...")
        # came from a check here. That check is now effectively covered by `if not sitemap_entries_from_module:`
        # If we reach this point, processed_for_csv is guaranteed to be non-empty.

        os.makedirs(site_specific_output_path, exist_ok=True)
        with open(sitemap_csv_full_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["normalized_url", "lastmod_iso"]) # CSV Header
            writer.writerows(processed_for_csv) # CSV Data
        
        sitemap_results.update({
            "status": "success", 
            "sitemap_csv_path": str(sitemap_csv_full_path), 
            # This count will be the same as total_sitemap_entries_returned_by_crawler_module
            "sitemap_urls_found_with_valid_lastmod": len(processed_for_csv) 
        })
        logger.info(f"Sitemap: Sitemap data for {effective_site_url} saved to {sitemap_csv_full_path} with {len(processed_for_csv)} entries.")
    except Exception as e:
        logger.error(f"Sitemap: Error during sitemap processing for {effective_site_url}: {e}", exc_info=True)
        sitemap_results.update({"status": "error", "error": str(e)})
    return sitemap_results

@router.post("/start_crawl") # Replaced eel.expose
async def start_crawl_process(
    request: StartCrawlRequest
):
    """
    Initiates the web crawling process for multiple URLs using FastAPI.
    This function replaces the functionality of the /crawl_single_url/ FastAPI endpoint
    and handles batch processing of URLs from the remote server.
    """
    global _is_processing
    _is_processing = True
    
    urls = request.urls
    output_dir = request.output_dir
    max_concurrency = request.max_concurrency
    max_depth = request.max_depth

    overall_results = {
        "status": "processing",
        "total_urls_to_crawl": len(urls),
        "output_base_directory": os.path.abspath(output_dir),
        "per_url_results": {}
    }

    # Ensure the base output directory exists
    try:
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    except Exception as e:
        _is_processing = False
        raise HTTPException(status_code=500, detail=f"Failed to create base output directory {output_dir}: {str(e)}")

    for i, url_original_input in enumerate(urls):
        logger.info(f"\n--- Processing URL {i+1}/{len(urls)}: {url_original_input} ---")
        single_url_crawl_summary = {"initial_url_input": url_original_input}

        try:
            schemed_url = prepare_initial_url_scheme(url_original_input)
            if not urlparse(schemed_url).netloc:
                single_url_crawl_summary.update({
                    "status": "skipped", 
                    "message": f"Invalid URL structure after preparation: '{url_original_input}' became '{schemed_url}'. Must have a domain."
                })
                overall_results["per_url_results"][url_original_input] = single_url_crawl_summary
                continue
            
            site_results = await crawl_website_single_site(
                start_url_original_schemed=schemed_url,
                output_dir=output_dir,
                max_concurrency=max_concurrency,
                max_depth=max_depth
            )
            single_url_crawl_summary.update(site_results)

            original_site_content_path = site_results.get("output_path_for_site")
            effective_url_for_sitemap = site_results.get("effective_start_url")

            sitemap_processing_summary = None
            if original_site_content_path and os.path.isdir(original_site_content_path):
                if effective_url_for_sitemap:
                    logger.info(f"Sitemap: Initiating sitemap processing for {effective_url_for_sitemap} into {original_site_content_path}")
                    sitemap_processing_summary = await process_and_save_sitemap(effective_url_for_sitemap, original_site_content_path)
                else:
                    err_msg_sitemap = "Sitemap skipped: no effective URL to process."
                    logger.warning(f"Sitemap: Skipping for (original input {url_original_input}): {err_msg_sitemap}")
                    sitemap_processing_summary = {"status": "skipped", "error": err_msg_sitemap}
                single_url_crawl_summary["sitemap_processing_results"] = sitemap_processing_summary

                metadata_filename = "crawl_metadata.json"
                site_folder_name_in_zip = os.path.basename(original_site_content_path)
                relative_metadata_path_in_zip = os.path.join(site_folder_name_in_zip, metadata_filename)
                
                # Create a snapshot of the summary *as it would be* if stored in the metadata file within the zip
                summary_for_metadata_file = {
                    key: value for key, value in single_url_crawl_summary.items()
                    if key not in ["site_output_zip_file", "data_folder_deleted_after_zip"]
                }
                summary_for_metadata_file["metadata_file_info_in_archive"] = (
                    f"This metadata is stored as '{relative_metadata_path_in_zip}' "
                    f"(relative to the root of the unzipped archive)."
                )

                metadata_full_path_on_disk = Path(original_site_content_path) / metadata_filename
                try:
                    metadata_content_to_save = {
                        "crawl_parameters": {
                            "requested_url_original": url_original_input,
                            "requested_url_schemed": schemed_url,
                            "base_output_directory_target_for_zip": os.path.abspath(output_dir),
                            "max_concurrency": max_concurrency,
                            "max_depth": max_depth
                        },
                        "crawl_summary_snapshot": summary_for_metadata_file
                    }
                    with open(metadata_full_path_on_disk, "w", encoding="utf-8") as f:
                        json.dump(metadata_content_to_save, f, indent=2, ensure_ascii=False)
                    logger.info(f"Metadata for single URL crawl saved to: {metadata_full_path_on_disk} (will be included in ZIP)")
                    single_url_crawl_summary["metadata_file_location_note"] = (
                        f"'{relative_metadata_path_in_zip}' (relative to unzipped archive root). "
                        f"Originally at {metadata_full_path_on_disk} before zipping."
                    )
                except Exception as e:
                    logger.error(f"Error saving metadata to {metadata_full_path_on_disk}: {e}")
                    single_url_crawl_summary["metadata_save_error"] = str(e)
                    single_url_crawl_summary["metadata_file_location_note"] = f"Failed to save metadata to {metadata_full_path_on_disk}."

                zip_base_name = os.path.join(output_dir, f"{site_folder_name_in_zip}_output")

                logger.info(f"Attempting to zip site output: {original_site_content_path} into {zip_base_name}.zip")
                zip_file_path_for_site = create_zip_archive(original_site_content_path, zip_base_name)

                if zip_file_path_for_site:
                    single_url_crawl_summary["site_output_zip_file"] = zip_file_path_for_site
                    logger.info(f"Output for {url_original_input} zipped to {zip_file_path_for_site}")
                    single_url_crawl_summary["metadata_file_location_note"] = (
                         f"'{relative_metadata_path_in_zip}' (relative to unzipped archive root) is inside {zip_file_path_for_site}."
                    )

                    try:
                        shutil.rmtree(original_site_content_path)
                        logger.info(f"Successfully deleted original content folder: {original_site_content_path}")
                        single_url_crawl_summary["data_folder_deleted_after_zip"] = True
                    except Exception as e:
                        logger.error(f"Failed to delete original content folder {original_site_content_path} after zipping: {e}", exc_info=True)
                        single_url_crawl_summary["data_folder_deleted_after_zip"] = False
                        single_url_crawl_summary["delete_folder_error"] = str(e)
                else:
                    single_url_crawl_summary["site_output_zip_file_error"] = f"Failed to zip {original_site_content_path}"
                    logger.error(f"Failed to zip output for {url_original_input}. Original folder {original_site_content_path} will remain.")
                    single_url_crawl_summary["metadata_file_location_note"] = (
                        f"Metadata at {metadata_full_path_on_disk} (Zipping failed, folder not deleted)."
                    )
            else:
                error_msg = (f"Output path for site data ('{original_site_content_path}') was invalid or not created. "
                             f"Cannot proceed with sitemap, metadata, or zipping for this site.")
                logger.error(error_msg)
                current_errors = single_url_crawl_summary.get("errors", [])
                if not isinstance(current_errors, list): current_errors = [str(current_errors)]
                current_errors.append(error_msg)
                single_url_crawl_summary["errors"] = current_errors

                if not single_url_crawl_summary.get("sitemap_processing_results"):
                     single_url_crawl_summary["sitemap_processing_results"] = {"status": "skipped", "error": "Prerequisite site output path invalid."}
                single_url_crawl_summary["metadata_file_location_note"] = "Skipped due to invalid site output path."

        except Exception as e:
            logger.critical(f"Critical error during crawl of {url_original_input}: {e}", exc_info=True)
            single_url_crawl_summary.update({"status": "error", "message": f"Internal crawl error: {str(e)}"})
        finally:
            overall_results["per_url_results"][url_original_input] = single_url_crawl_summary
    
    overall_results["status"] = "completed"
    logger.info(f"--- Overall crawl process completed for {len(urls)} URLs ---")
    _is_processing = False # Set to idle after completion
    return overall_results

@router.get("/server_status") # Replaced eel.expose
def get_server_status():
    """
    Returns the current processing status of the server (idle or in use).
    """
    return {"status": "processing" if _is_processing else "idle"}

@router.post("/notify_finished") # Replaced eel.expose
async def notify_server_crawl_finished(request: NotifyFinishedRequest):
    """
    Notifies the master server that this client machine has finished a crawl process.
    This sends a comprehensive status update, including the outcome of the crawl.
    """
    master_server_ip = request.master_server_ip
    master_server_port = request.master_server_port
    processed_successfully = request.processed_successfully

    target_url = f"http://{master_server_ip}:{master_server_port}/update_machine_status"

    # Get current local machine metrics to send a full status update
    metrics_result = get_local_machine_metrics()
    if metrics_result["status"] == "error":
        logger.error(f"Failed to get local machine metrics for final status update: {metrics_result["message"]}")
        # Proceed with a minimal payload if metrics can't be retrieved
        machine_name = platform.node() + "_unknown"
        current_crawling_status = "finished_with_issues" # Assume issues if metrics failed
    else:
        machine_name = metrics_result["machine_name"]
        current_crawling_status = "finished_successfully" if processed_successfully else "finished_with_issues"

    # Prepare the payload with comprehensive machine status and crawl outcome
    payload = {
        "machine_name": machine_name,
        "total_storage_gb": metrics_result.get("total_storage_gb", 0.0),
        "free_storage_gb": metrics_result.get("free_storage_gb", 0.0),
        "cpu_usage_percent": metrics_result.get("cpu_usage_percent", 0.0),
        "memory_usage_percent": metrics_result.get("memory_usage_percent", 0.0),
        "crawling_status": current_crawling_status
    }
    
    logger.info(f"Notifying master server {master_server_ip}:{master_server_port} that processing is finished (success: {processed_successfully})...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(target_url, json=payload, timeout=5) as response:
                response_data = await response.json()
                if response.status == 200:
                    logger.info(f"Master server {master_server_ip}:{master_server_port} acknowledged completion: {response_data.get('message', 'Unknown response')}")
                    return {"status": "acknowledged", "message": response_data.get('message', 'Master acknowledged completion.')}
                else:
                    error_text = response_data.get('detail', await response.text())
                    logger.warning(f"Master server {master_server_ip}:{master_server_port} responded with {response.status} when notifying completion: {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Master did not acknowledge completion: {error_text}")
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Could not connect to master server {master_server_ip}:{master_server_port} to notify completion: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to master server ({master_server_ip}:{master_server_port}) to notify completion: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout when notifying master server {master_server_ip}:{master_server_port} of completion.")
        raise HTTPException(status_code=504, detail=f"Timeout when notifying master server ({master_server_ip}:{master_server_port}) of completion.")
    except Exception as e:
        logger.error(f"Unexpected error when notifying master server {master_server_ip}:{master_server_port} of completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error notifying master server: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    app.include_router(router) # Include the router in the app
    uvicorn.run(app, host="0.0.0.0", port=8000)