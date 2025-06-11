import shutil
import asyncio
import os
import json
import re
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from typing import List, Dict, Any, Tuple, Optional, Set
import csv
import io
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
import platform
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import logging
import sys
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app and router
app = FastAPI()
router = APIRouter()

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
EXCLUDE_KEYWORDS = ['pdf', 'jpeg', 'jpg', 'png', 'webp']
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

# --- Sitemap Crawler Import ---
try:
    from sitemap_crawler import get_sitemap_data_for_single_url
    SITEMAP_CRAWLER_AVAILABLE = True
    logger.info("Successfully imported sitemap_crawler module.")
except ImportError:
    SITEMAP_CRAWLER_AVAILABLE = False
    logger.warning("sitemap_crawler.py not found. Sitemap processing will be skipped.")
    async def get_sitemap_data_for_single_url(*args, **kwargs) -> List[Tuple[str, str]]:
        logger.error("Sitemap crawler module was not imported correctly. Cannot get sitemap data.")
        return []

# --- URL Utilities ---
def prepare_initial_url_scheme(url_str: str) -> str:
    if not url_str: return ""
    url_str = url_str.strip()
    parsed = urlparse(url_str)
    if not parsed.scheme:
        return f"http://{url_str.lstrip('//')}"
    return url_str

@router.get("/metrics")
def get_local_machine_metrics():
    try:
        machine_name = platform.node()
        total_disk_gb, free_disk_gb = 0, 0
        for part in psutil.disk_partitions(all=False):
            if os.name == 'nt' and ('cdrom' in part.opts or part.fstype == ''):
                continue
            usage = psutil.disk_usage(part.mountpoint)
            total_disk_gb += usage.total / (1024**3)
            free_disk_gb += usage.free / (1024**3)
        return {
            "status": "success",
            "machine_name": machine_name,
            "total_storage_gb": round(total_disk_gb, 2),
            "free_storage_gb": round(free_disk_gb, 2),
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "crawling_status": "in_use" if _is_processing else "idle"
        }
    except Exception as e:
        logger.error(f"Error getting local machine metrics: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@router.post("/fetch_urls")
async def fetch_urls_from_master_server(request: FetchUrlsRequest):
    target_url = f"http://{request.master_server_ip}:{request.master_server_port}/get_urls_to_crawl"
    logger.info(f"Attempting to fetch URLs from master server: {target_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target_url, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                logger.info(f"Fetched {len(data.get('urls', []))} URLs from master.")
                return data
    except Exception as e:
        logger.error(f"Failed to fetch URLs from master: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send_status")
async def send_local_machine_status_to_master(request: SendStatusRequest):
    metrics = get_local_machine_metrics()
    if metrics["status"] == "error":
        raise HTTPException(status_code=500, detail="Failed to get local machine metrics.")
    target_url = f"http://{request.master_server_ip}:{request.master_server_port}/update_machine_status"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(target_url, json=metrics, timeout=5) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Failed to send status to master: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/machines_status")
async def get_all_machines_status_from_master(master_server_ip: str, master_server_port: int):
    target_url = f"http://{master_server_ip}:{master_server_port}/get_all_machines_status"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(target_url, timeout=5) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Failed to get machine statuses from master: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper Functions ---
def create_zip_archive(source_dir_path: str, output_zip_base_name: str) -> Optional[str]:
    try:
        source_dir_abs = os.path.abspath(source_dir_path)
        root_dir = os.path.dirname(source_dir_abs)
        base_dir_to_zip = os.path.basename(source_dir_abs)
        logger.info(f"Creating ZIP: output_base='{output_zip_base_name}', root_dir='{root_dir}', base_dir='{base_dir_to_zip}'")
        zip_file_path = shutil.make_archive(base_name=output_zip_base_name, format='zip', root_dir=root_dir, base_dir=base_dir_to_zip)
        logger.info(f"Successfully created ZIP archive: {zip_file_path}")
        return zip_file_path
    except Exception as e:
        logger.error(f"Error creating ZIP archive for {source_dir_path}: {e}", exc_info=True)
        return None

def normalize_url_for_deduplication(url_string: str) -> str:
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
        logger.warning(f"Could not normalize URL '{url_string}': {e}. Returning original.")
        return url_string.rstrip('/')

async def resolve_initial_url(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        async with session.get(url, allow_redirects=True, timeout=20) as response:
            if response.status >= 400:
                msg = f"HTTP {response.status} for {response.url}"
                logger.warning(f"Initial URL resolution failed for '{url}': {msg}")
                return None, msg
            logger.info(f"URL '{url}' resolved to '{response.url}'")
            return str(response.url), None
    except Exception as e:
        logger.error(f"Error resolving initial URL '{url}': {e}", exc_info=True)
        return None, str(e)

def clean_markdown(md_text: str) -> str:
    md_text = re.sub(r'!\[([^]]*)\]\((http[s]?://[^)]+)\)', '', md_text)
    md_text = re.sub(r'\[([^]]+)\]\((http[s]?://[^)]+)\)', r'\1', md_text)
    md_text = re.sub(r'\n[ \t]*\n+', '\n\n', md_text)
    return md_text.strip()

def sanitize_filename(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace(".", "_")
        path = parsed.path.strip("/").replace("/", "_").replace(".", "_") if parsed.path else "index"
        query_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', parsed.query[:50]) if parsed.query else ""
        filename_base = f"{netloc}_{path}_{query_safe}".strip('_')
        filename_base = re.sub(r'_+', '_', filename_base)
        return filename_base[:250] + ".md"
    except Exception:
        return f"url_{abs(hash(url))}.md"

def sanitize_dirname(url: str) -> str:
    try:
        return re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(url).netloc)
    except Exception:
        return f"domain_{abs(hash(url))}"

CrawlQueueItem = Tuple[str, int, str, str]

def process_markdown_and_save(url: str, markdown_content: str, output_path: str) -> Dict[str, Any]:
    try:
        cleaned_markdown = clean_markdown(markdown_content)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Original URL (effective for crawl): {url}\n\n{cleaned_markdown}\n")
        logger.info(f"Saved cleaned Markdown to: {output_path}")
        return {"status": "success", "url": url, "path": output_path}
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
    
    async with aiohttp.ClientSession(headers=COMMON_HEADERS) as http_session:
        effective_start_url, error_msg = await resolve_initial_url(http_session, start_url_original_schemed)
        if not effective_start_url:
            results["failed"].append({"url": start_url_original_schemed, "error": f"Resolution failed: {error_msg}"})
            return results

    results["effective_start_url"] = effective_start_url
    try:
        parsed_effective_url = urlparse(effective_start_url)
        effective_start_domain = parsed_effective_url.netloc
        if not effective_start_domain:
            raise ValueError(f"Could not extract domain from effective URL: {effective_start_url}")
        
        site_subdir_name = sanitize_dirname(effective_start_url)
        site_output_path_specific = os.path.abspath(os.path.join(output_dir, site_subdir_name))
        os.makedirs(site_output_path_specific, exist_ok=True)
        results["output_path_for_site"] = site_output_path_specific
        logger.info(f"Effective domain: {effective_start_domain}, Output path: {site_output_path_specific}")

    except Exception as e:
        results["failed"].append({"url": start_url_original_schemed, "error": f"Directory setup failed: {e}"})
        logger.error(f"Directory setup failed for {start_url_original_schemed}: {e}", exc_info=True)
        return results

    # We provide NO filtering arguments to the library. We will do it ourselves.
    md_generator = DefaultMarkdownGenerator(options={"ignore_links": True, "escape_html": True, "body_width": 0})
    config = CrawlerRunConfig(
        markdown_generator=md_generator,
        cache_mode="BYPASS",
        exclude_social_media_links=True
    )

    crawled_urls: Set[str] = set()
    queued_urls: Set[str] = set()
    crawl_queue: asyncio.Queue[CrawlQueueItem] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrency)

    initial_normalized_url = normalize_url_for_deduplication(effective_start_url)
    crawl_queue.put_nowait((initial_normalized_url, 0, effective_start_domain, site_output_path_specific))
    queued_urls.add(initial_normalized_url)
    
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        async def crawl_page_worker():
            while True:
                try:
                    current_url, current_depth, expected_domain, current_path = await asyncio.wait_for(crawl_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if crawl_queue.empty(): break
                    continue
                except asyncio.CancelledError:
                    break
                
                if current_url in crawled_urls:
                    crawl_queue.task_done()
                    continue
                
                crawled_urls.add(current_url)
                logger.info(f"Crawling ({len(crawled_urls)}/{len(queued_urls)}): {current_url} (Depth: {current_depth})")
                
                try:
                    async with semaphore:
                        async with AsyncWebCrawler(verbose=False) as crawler:
                            page_data = await crawler.arun(url=current_url, config=config)

                    final_url_processed = normalize_url_for_deduplication(page_data.url or current_url)

                    if not page_data.success:
                        logger.warning(f"Crawl failed for {current_url}: {page_data.error_message}")
                        results["failed"].append({"url": current_url, "error": page_data.error_message})
                        continue

                    if page_data.markdown:
                        filename = sanitize_filename(final_url_processed)
                        output_path = os.path.join(current_path, filename)
                        loop = asyncio.get_running_loop()
                        process_result = await loop.run_in_executor(
                            executor, process_markdown_and_save, final_url_processed, page_data.markdown.raw_markdown, output_path
                        )
                        if process_result["status"] == "success":
                            results["success"].append(final_url_processed)
                        else:
                            results["failed"].append({"url": final_url_processed, "error": process_result["error"]})
                    else:
                        logger.warning(f"No markdown content for {final_url_processed}")

                    # --- MANUAL LINK FILTERING LOGIC ---
                    if current_depth < max_depth and page_data.links:
                        # Combine all links, as we can't trust the library's categorization
                        all_links = page_data.links.get("internal", []) + page_data.links.get("external", [])
                        
                        for link_info in all_links:
                            href = link_info.get("href")
                            if not href:
                                continue
                            
                            try:
                                absolute_url = urljoin(final_url_processed, href)
                                normalized_url = normalize_url_for_deduplication(absolute_url)
                                
                                # Our manual domain check
                                if urlparse(normalized_url).netloc == expected_domain:
                                    if normalized_url not in crawled_urls and normalized_url not in queued_urls:
                                        queued_urls.add(normalized_url)
                                        await crawl_queue.put((normalized_url, current_depth + 1, expected_domain, current_path))
                            except Exception as link_e:
                                logger.warning(f"Link processing error for '{href}': {link_e}")
                
                except Exception as e_crawl:
                    logger.error(f"Critical worker error for {current_url}: {e_crawl}", exc_info=True)
                    results["failed"].append({"url": current_url, "error": f"Worker exception: {e_crawl}"})
                finally:
                    crawl_queue.task_done()

        worker_tasks = [asyncio.create_task(crawl_page_worker()) for _ in range(max_concurrency)]
        await crawl_queue.join()
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    logger.info(f"Finished crawl for {start_url_original_schemed}")
    return results

async def process_and_save_sitemap(effective_url: str, output_path: str) -> Dict[str, Any]:
    sitemap_results = {"status": "initiated", "based_on_effective_url": effective_url}
    if not SITEMAP_CRAWLER_AVAILABLE:
        sitemap_results.update({"status": "skipped", "error": "Sitemap crawler module not available."})
        return sitemap_results
    
    try:
        async with aiohttp.ClientSession(headers=COMMON_HEADERS) as session:
            sitemap_entries = await get_sitemap_data_for_single_url(effective_url, session)
        
        sitemap_results["total_sitemap_entries_returned_by_crawler_module"] = len(sitemap_entries)
        if not sitemap_entries:
            sitemap_results.update({"status": "no_valid_sitemap_data_found_by_module", "error": "No entries with valid lastmod found."})
            return sitemap_results

        csv_path = Path(output_path) / "sitemap_data.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["normalized_url", "lastmod_iso"])
            writer.writerows(sitemap_entries)
        
        sitemap_results.update({
            "status": "success",
            "sitemap_csv_path": str(csv_path),
            "sitemap_urls_found_with_valid_lastmod": len(sitemap_entries)
        })
    except Exception as e:
        logger.error(f"Sitemap processing error for {effective_url}: {e}", exc_info=True)
        sitemap_results.update({"status": "error", "error": str(e)})
    return sitemap_results

@router.post("/start_crawl")
async def start_crawl_process(request: StartCrawlRequest):
    global _is_processing
    if _is_processing:
        raise HTTPException(status_code=429, detail="A crawl is already in progress.")
    _is_processing = True

    overall_results = {"per_url_results": {}}
    try:
        os.makedirs(os.path.abspath(request.output_dir), exist_ok=True)
        for url_input in request.urls:
            logger.info(f"\n--- Processing URL: {url_input} ---")
            schemed_url = prepare_initial_url_scheme(url_input)
            
            crawl_summary = await crawl_website_single_site(
                start_url_original_schemed=schemed_url,
                output_dir=request.output_dir,
                max_concurrency=request.max_concurrency,
                max_depth=request.max_depth
            )
            
            site_path = crawl_summary.get("output_path_for_site")
            effective_url = crawl_summary.get("effective_start_url")

            if site_path and effective_url and os.path.isdir(site_path):
                sitemap_summary = await process_and_save_sitemap(effective_url, site_path)
                crawl_summary["sitemap_processing_results"] = sitemap_summary

                # Metadata and Zipping
                metadata_path = Path(site_path) / "crawl_metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump({"crawl_summary_snapshot": crawl_summary}, f, indent=2, ensure_ascii=False)
                
                zip_base_name = os.path.join(request.output_dir, f"{os.path.basename(site_path)}_output")
                zip_file = create_zip_archive(site_path, zip_base_name)
                
                if zip_file:
                    crawl_summary["site_output_zip_file"] = zip_file
                    try:
                        shutil.rmtree(site_path)
                        crawl_summary["data_folder_deleted_after_zip"] = True
                    except Exception as e:
                        logger.error(f"Failed to delete folder {site_path}: {e}")
                        crawl_summary["data_folder_deleted_after_zip"] = False
                else:
                    logger.error(f"Failed to zip folder {site_path}")

            overall_results["per_url_results"][url_input] = crawl_summary

    except Exception as e:
        logger.critical(f"Overall crawl process failed: {e}", exc_info=True)
        _is_processing = False
        raise HTTPException(status_code=500, detail=f"Crawl process failed: {e}")
    
    _is_processing = False
    return overall_results

@router.get("/server_status")
def get_server_status():
    return {"status": "processing" if _is_processing else "idle"}

@router.post("/notify_finished")
async def notify_server_crawl_finished(request: NotifyFinishedRequest):
    payload = get_local_machine_metrics()
    if payload['status'] == 'error':
        raise HTTPException(status_code=500, detail="Failed to get machine metrics.")
        
    payload["crawling_status"] = "finished_successfully" if request.processed_successfully else "finished_with_issues"
    target_url = f"http://{request.master_server_ip}:{request.master_server_port}/update_machine_status"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(target_url, json=payload, timeout=5) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Failed to notify master of completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)