# import shutil
# import asyncio
# import os
# import json
# import re
# import time
# from urllib.parse import urljoin, urlparse, urlunparse
# from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
# from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
# from typing import List, Dict, Any, Tuple, Optional, Set
# import csv
# import io
# from concurrent.futures import ThreadPoolExecutor
# import aiohttp
# import psutil
# import platform
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, ValidationError
# import logging
# import sys
# from pathlib import Path

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(title="Crawling Worker WebSocket API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*'],
#     allow_credentials=True,
#     allow_methods=['*'],
#     allow_headers=['*'],
# )

# # Force UTF-8 encoding at the top of the script
# if sys.platform.startswith('win'):
#     os.environ['PYTHONIOENCODING'] = 'utf-8'
#     sys.stdout.reconfigure(encoding='utf-8')
#     sys.stderr.reconfigure(encoding='utf-8')

# # --- Constants ---
# COMMON_HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
#     "Accept-Language": "en-US,en;q=0.9",
# }
# EXCLUDE_KEYWORDS = ['pdf', 'jpeg', 'jpg', 'png', 'webp']
# _is_processing = False

# # --- Pydantic Models for WebSocket Message Payloads ---
# class StartCrawlPayload(BaseModel):
#     urls: List[str]
#     output_dir: str
#     max_concurrency: int = 8
#     max_depth: int = 2

# # --- Sitemap Crawler Import ---
# try:
#     from sitemap_crawler import get_sitemap_data_for_single_url
#     SITEMAP_CRAWLER_AVAILABLE = True
#     logger.info("Successfully imported sitemap_crawler module.")
# except ImportError:
#     SITEMAP_CRAWLER_AVAILABLE = False
#     logger.warning("sitemap_crawler.py not found. Sitemap processing will be skipped.")
#     async def get_sitemap_data_for_single_url(*args, **kwargs) -> List[Tuple[str, str]]:
#         logger.error("Sitemap crawler module was not imported correctly. Cannot get sitemap data.")
#         return []
        
# # --- WebSocket Connection Manager ---
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []
#         # Modified: Added connection_states to track per-connection crawl state
#         self.connection_states: Dict[WebSocket, bool] = {}

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)
#         self.connection_states[websocket] = False
#         logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)
#         self.connection_states.pop(websocket, None)  # Modified: Remove connection state
#         logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

#     async def send_personal_message(self, message: dict, websocket: WebSocket):
#         try:
#             await websocket.send_json(message)
#         except Exception as e:
#             logger.warning(f"Failed to send message to a client: {e}")

# manager = ConnectionManager()


# # --- Worker-as-Client Functions (Conceptual for WebSocket) ---
# # NOTE: The following functions would be part of a single, persistent WebSocket
# # connection to the master server, running in a background task.
# async def connect_to_master_and_manage_tasks(master_ip: str, master_port: int):
#     """
#     This function would replace all individual HTTP calls to the master.
#     It would establish a persistent WebSocket connection and handle both
#     sending status and receiving crawl jobs.
#     """
#     ws_url = f"ws://{master_ip}:{master_port}/ws_master"
#     logger.info(f"Concept: This is where the worker would connect to master at {ws_url}")
#     # In a real implementation:
#     # 1. Use a library like 'websockets' to connect to the master.
#     # 2. Start a loop to listen for messages (e.g., a "start_crawl" command from the master).
#     # 3. Start a separate concurrent task to periodically send status updates (get_local_machine_metrics)
#     #    over the same WebSocket connection.
#     # 4. Implement reconnect logic.
#     pass

# # --- URL Utilities & Core Logic (largely unchanged) ---
# def get_local_machine_metrics():
#     try:
#         machine_name = platform.node()
#         total_disk_gb, free_disk_gb = 0, 0
#         for part in psutil.disk_partitions(all=False):
#             if os.name == 'nt' and ('cdrom' in part.opts or part.fstype == ''):
#                 continue
#             usage = psutil.disk_usage(part.mountpoint)
#             total_disk_gb += usage.total / (1024**3)
#             free_disk_gb += usage.free / (1024**3)
#         return {
#             "status": "success",
#             "machine_name": machine_name,
#             "total_storage_gb": round(total_disk_gb, 2),
#             "free_storage_gb": round(free_disk_gb, 2),
#             "cpu_usage_percent": psutil.cpu_percent(interval=1),
#             "memory_usage_percent": psutil.virtual_memory().percent,
#             "crawling_status": "in_use" if _is_processing else "idle"
#         }
#     except Exception as e:
#         logger.error(f"Error getting local machine metrics: {e}", exc_info=True)
#         return {"status": "error", "message": str(e)}

# def prepare_initial_url_scheme(url_str: str) -> str:
#     if not url_str: return ""
#     url_str = url_str.strip()
#     parsed = urlparse(url_str)
#     if not parsed.scheme:
#         return f"http://{url_str.lstrip('//')}"
#     return url_str

# def create_zip_archive(source_dir_path: str, output_zip_base_name: str) -> Optional[str]:
#     try:
#         source_dir_abs = os.path.abspath(source_dir_path)
#         root_dir = os.path.dirname(source_dir_abs)
#         base_dir_to_zip = os.path.basename(source_dir_abs)
#         logger.info(f"Creating ZIP: output_base='{output_zip_base_name}', root_dir='{root_dir}', base_dir='{base_dir_to_zip}'")
#         zip_file_path = shutil.make_archive(base_name=output_zip_base_name, format='zip', root_dir=root_dir, base_dir=base_dir_to_zip)
#         logger.info(f"Successfully created ZIP archive: {zip_file_path}")
#         return zip_file_path
#     except Exception as e:
#         logger.error(f"Error creating ZIP archive for {source_dir_path}: {e}", exc_info=True)
#         return None

# def normalize_url_for_deduplication(url_string: str) -> str:
#     try:
#         parsed = urlparse(url_string)
#         path = parsed.path
#         if path and not path.startswith('/'):
#             path = '/' + path
#         elif not path and (parsed.query or parsed.params):
#             path = '/'
#         return urlunparse((
#             parsed.scheme.lower(),
#             parsed.netloc.lower(),
#             path or '/',
#             parsed.params,
#             parsed.query,
#             ''
#         )).rstrip('/')
#     except Exception as e:
#         logger.warning(f"Could not normalize URL '{url_string}': {e}. Returning original.")
#         return url_string.rstrip('/')

# async def resolve_initial_url(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], Optional[str]]:
#     try:
#         async with session.get(url, allow_redirects=True, timeout=20) as response:
#             if response.status >= 400:
#                 msg = f"HTTP {response.status} for {response.url}"
#                 logger.warning(f"Initial URL resolution failed for '{url}': {msg}")
#                 return None, msg
#             logger.info(f"URL '{url}' resolved to '{response.url}'")
#             return str(response.url), None
#     except Exception as e:
#         logger.error(f"Error resolving initial URL '{url}': {e}", exc_info=True)
#         return None, str(e)

# def clean_markdown(md_text: str) -> str:
#     md_text = re.sub(r'!\[([^]]*)\]\((http[s]?://[^)]+)\)', '', md_text)
#     md_text = re.sub(r'\[([^]]+)\]\((http[s]?://[^)]+)\)', r'\1', md_text)
#     md_text = re.sub(r'\n[ \t]*\n+', '\n\n', md_text)
#     return md_text.strip()

# def sanitize_filename(url: str) -> str:
#     try:
#         parsed = urlparse(url)
#         netloc = parsed.netloc.replace(".", "_")
#         path = parsed.path.strip("/").replace("/", "_").replace(".", "_") if parsed.path else "index"
#         query_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', parsed.query[:50]) if parsed.query else ""
#         filename_base = f"{netloc}_{path}_{query_safe}".strip('_')
#         filename_base = re.sub(r'_+', '_', filename_base)
#         return filename_base[:250] + ".md"
#     except Exception:
#         return f"url_{abs(hash(url))}.md"

# def sanitize_dirname(url: str) -> str:
#     try:
#         return re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(url).netloc)
#     except Exception:
#         return f"domain_{abs(hash(url))}"

# CrawlQueueItem = Tuple[str, int, str, str]

# def process_markdown_and_save(url: str, markdown_content: str, output_path: str) -> Dict[str, Any]:
#     try:
#         cleaned_markdown = clean_markdown(markdown_content)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(f"# Original URL (effective for crawl): {url}\n\n{cleaned_markdown}\n")
#         logger.info(f"Saved cleaned Markdown to: {output_path}")
#         return {"status": "success", "url": url, "path": output_path}
#     except Exception as e:
#         logger.error(f"Error processing/saving Markdown for {url} to {output_path}: {e}", exc_info=True)
#         return {"status": "failed", "url": url, "error": str(e)}

# async def crawl_website_single_site(
#     start_url_original_schemed: str,
#     output_dir: str,
#     max_concurrency: int,
#     max_depth: int,
#     websocket: Optional[WebSocket] = None # Added for real-time progress
# ) -> Dict[str, Any]:
#     results: Dict[str, Any] = {
#         "success": [], "failed": [], "skipped_by_filter": [],
#         "initial_url": start_url_original_schemed,
#         "effective_start_url": None,
#         "output_path_for_site": None
#     }
    

#     # Modified: Throttled progress updates
#     async def send_progress(message: str, detail: Optional[Dict] = None, throttle_interval: float = 1.0):
#         if not hasattr(send_progress, 'last_sent'):
#             send_progress.last_sent = 0
#         current_time = time()
#         if websocket and (current_time - send_progress.last_sent >= throttle_interval):
#             payload = {"type": "crawl_progress", "message": message, "detail": detail or {}}
#             await manager.send_personal_message(payload, websocket)
#             send_progress.last_sent = current_time

#     await send_progress(f"Resolving initial URL: {start_url_original_schemed}")
#     async with aiohttp.ClientSession(headers=COMMON_HEADERS) as http_session:
#         effective_start_url, error_msg = await resolve_initial_url(http_session, start_url_original_schemed)
#         if not effective_start_url:
#             results["failed"].append({"url": start_url_original_schemed, "error": f"Resolution failed: {error_msg}"})
#             await send_progress(f"URL resolution failed: {error_msg}")
#             return results

#     results["effective_start_url"] = effective_start_url
#     await send_progress(f"URL resolved to: {effective_start_url}")
    
#     try:
#         parsed_effective_url = urlparse(effective_start_url)
#         effective_start_domain = parsed_effective_url.netloc
#         if not effective_start_domain:
#             raise ValueError(f"Could not extract domain from effective URL: {effective_start_url}")
        
#         site_subdir_name = sanitize_dirname(effective_start_url)
#         site_output_path_specific = os.path.abspath(os.path.join(output_dir, site_subdir_name))
#         os.makedirs(site_output_path_specific, exist_ok=True)
#         results["output_path_for_site"] = site_output_path_specific
#         logger.info(f"Effective domain: {effective_start_domain}, Output path: {site_output_path_specific}")
#         await send_progress(f"Output directory set to: {site_output_path_specific}")

#     except Exception as e:
#         results["failed"].append({"url": start_url_original_schemed, "error": f"Directory setup failed: {e}"})
#         logger.error(f"Directory setup failed for {start_url_original_schemed}: {e}", exc_info=True)
#         await send_progress(f"Error setting up directory: {e}")
#         return results

#     md_generator = DefaultMarkdownGenerator(options={"ignore_links": True, "escape_html": True, "body_width": 0})
#     config = CrawlerRunConfig(markdown_generator=md_generator, cache_mode="BYPASS", exclude_social_media_links=True)

#     crawled_urls: Set[str] = set()
#     queued_urls: Set[str] = set()
#     crawl_queue: asyncio.Queue[CrawlQueueItem] = asyncio.Queue()
#     semaphore = asyncio.Semaphore(max_concurrency)

#     initial_normalized_url = normalize_url_for_deduplication(effective_start_url)
#     crawl_queue.put_nowait((initial_normalized_url, 0, effective_start_domain, site_output_path_specific))
#     queued_urls.add(initial_normalized_url)
    
#     with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
#         async def crawl_page_worker():
#             while True:
#                 try:
#                     current_url, current_depth, expected_domain, current_path = await asyncio.wait_for(crawl_queue.get(), timeout=1.0)
#                 except asyncio.TimeoutError:
#                     if crawl_queue.empty(): break
#                     continue
#                 except asyncio.CancelledError: break
                
#                 if current_url in crawled_urls:
#                     crawl_queue.task_done()
#                     continue
                
#                 crawled_urls.add(current_url)
#                 progress_msg = f"Crawling ({len(crawled_urls)}/{len(queued_urls)}): {current_url} (Depth: {current_depth})"
#                 logger.info(progress_msg)
#                 await send_progress(progress_msg, {"crawled_count": len(crawled_urls), "queued_count": len(queued_urls)})
                
#                 try:
#                     async with semaphore:
#                         async with AsyncWebCrawler(verbose=False) as crawler:
#                             page_data = await crawler.arun(url=current_url, config=config)

#                     final_url_processed = normalize_url_for_deduplication(page_data.url or current_url)

#                     if not page_data.success:
#                         logger.warning(f"Crawl failed for {current_url}: {page_data.error_message}")
#                         results["failed"].append({"url": current_url, "error": page_data.error_message})
#                         await send_progress(f"Crawl failed for {current_url}", {"error": page_data.error_message})
#                         continue

#                     if page_data.markdown:
#                         filename = sanitize_filename(final_url_processed)
#                         output_path = os.path.join(current_path, filename)
#                         loop = asyncio.get_running_loop()
#                         process_result = await loop.run_in_executor(
#                             executor, process_markdown_and_save, final_url_processed, page_data.markdown.raw_markdown, output_path)
                        
#                         if process_result["status"] == "success":
#                             results["success"].append(final_url_processed)
#                             await send_progress(f"Saved: {final_url_processed}", {"path": output_path})
#                         else:
#                             results["failed"].append({"url": final_url_processed, "error": process_result["error"]})
#                             await send_progress(f"Save failed for {final_url_processed}", {"error": process_result["error"]})
#                     else:
#                         logger.warning(f"No markdown content for {final_url_processed}")

#                     if current_depth < max_depth and page_data.links:
#                         all_links = page_data.links.get("internal", []) + page_data.links.get("external", [])
#                         for link_info in all_links:
#                             href = link_info.get("href")
#                             if not href: continue
#                             try:
#                                 absolute_url = urljoin(final_url_processed, href)
#                                 normalized_url = normalize_url_for_deduplication(absolute_url)
#                                 if urlparse(normalized_url).netloc == expected_domain:
#                                     if normalized_url not in crawled_urls and normalized_url not in queued_urls:
#                                         queued_urls.add(normalized_url)
#                                         await crawl_queue.put((normalized_url, current_depth + 1, expected_domain, current_path))
#                             except Exception as link_e:
#                                 logger.warning(f"Link processing error for '{href}': {link_e}")
                
#                 except Exception as e_crawl:
#                     logger.error(f"Critical worker error for {current_url}: {e_crawl}", exc_info=True)
#                     results["failed"].append({"url": current_url, "error": f"Worker exception: {e_crawl}"})
#                     await send_progress(f"Critical error for {current_url}", {"error": str(e_crawl)})
#                 finally:
#                     crawl_queue.task_done()

#         worker_tasks = [asyncio.create_task(crawl_page_worker()) for _ in range(max_concurrency)]
#         await crawl_queue.join()
#         for task in worker_tasks: task.cancel()
#         await asyncio.gather(*worker_tasks, return_exceptions=True)

#     logger.info(f"Finished crawl for {start_url_original_schemed}")
#     await send_progress(f"Finished crawl for {start_url_original_schemed}")
#     return results

# async def process_and_save_sitemap(effective_url: str, output_path: str, websocket: Optional[WebSocket] = None) -> Dict[str, Any]:
#     sitemap_results = {"status": "initiated", "based_on_effective_url": effective_url}
    
#     async def send_sitemap_progress(message: str, detail: Optional[Dict] = None):
#         if websocket:
#             payload = {"type": "sitemap_progress", "message": message, "detail": detail or {}}
#             await manager.send_personal_message(payload, websocket)
            
#     await send_sitemap_progress("Sitemap processing started.")
#     if not SITEMAP_CRAWLER_AVAILABLE:
#         error_msg = "Sitemap crawler module not available."
#         sitemap_results.update({"status": "skipped", "error": error_msg})
#         await send_sitemap_progress(f"Skipped: {error_msg}")
#         return sitemap_results
    
#     try:
#         async with aiohttp.ClientSession(headers=COMMON_HEADERS) as session:
#             sitemap_entries = await get_sitemap_data_for_single_url(effective_url, session)
        
#         sitemap_results["total_sitemap_entries_returned_by_crawler_module"] = len(sitemap_entries)
#         if not sitemap_entries:
#             error_msg = "No entries with valid lastmod found."
#             sitemap_results.update({"status": "no_valid_sitemap_data_found_by_module", "error": error_msg})
#             await send_sitemap_progress(f"No valid sitemap data found.")
#             return sitemap_results

#         csv_path = Path(output_path) / "sitemap_data.csv"
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(["normalized_url", "lastmod_iso"])
#             writer.writerows(sitemap_entries)
        
#         sitemap_results.update({"status": "success", "sitemap_csv_path": str(csv_path), "sitemap_urls_found_with_valid_lastmod": len(sitemap_entries)})
#         await send_sitemap_progress(f"Successfully saved {len(sitemap_entries)} sitemap URLs.", {"path": str(csv_path)})
#     except Exception as e:
#         logger.error(f"Sitemap processing error for {effective_url}: {e}", exc_info=True)
#         sitemap_results.update({"status": "error", "error": str(e)})
#         await send_sitemap_progress("Sitemap processing failed.", {"error": str(e)})
#     return sitemap_results

# async def run_crawl_and_notify(websocket: WebSocket, payload: StartCrawlPayload):
#     """A wrapper to run the full crawl process and manage state."""
#     global _is_processing
#     _is_processing = True
#     overall_results = {"per_url_results": {}}
    
#     try:
#         os.makedirs(os.path.abspath(payload.output_dir), exist_ok=True)
#         for url_input in payload.urls:
#             await manager.send_personal_message({"type": "info", "message": f"Starting processing for URL: {url_input}"}, websocket)
#             schemed_url = prepare_initial_url_scheme(url_input)
            
#             crawl_summary = await crawl_website_single_site(
#                 start_url_original_schemed=schemed_url,
#                 output_dir=payload.output_dir,
#                 max_concurrency=payload.max_concurrency,
#                 max_depth=payload.max_depth,
#                 websocket=websocket
#             )
            
#             site_path = crawl_summary.get("output_path_for_site")
#             effective_url = crawl_summary.get("effective_start_url")

#             if site_path and effective_url and os.path.isdir(site_path):
#                 sitemap_summary = await process_and_save_sitemap(effective_url, site_path, websocket)
#                 crawl_summary["sitemap_processing_results"] = sitemap_summary

#                 metadata_path = Path(site_path) / "crawl_metadata.json"
#                 with open(metadata_path, "w", encoding="utf-8") as f:
#                     json.dump({"crawl_summary_snapshot": crawl_summary}, f, indent=2, ensure_ascii=False)
                
#                 zip_base_name = os.path.join(payload.output_dir, f"{os.path.basename(site_path)}_output")
#                 zip_file = create_zip_archive(site_path, zip_base_name)
                
#                 if zip_file:
#                     crawl_summary["site_output_zip_file"] = zip_file
#                     await manager.send_personal_message({"type": "info", "message": f"Created ZIP archive: {zip_file}"}, websocket)
#                     try:
#                         shutil.rmtree(site_path)
#                         crawl_summary["data_folder_deleted_after_zip"] = True
#                     except Exception as e:
#                         logger.error(f"Failed to delete folder {site_path}: {e}")
#                         crawl_summary["data_folder_deleted_after_zip"] = False
#                 else:
#                     logger.error(f"Failed to zip folder {site_path}")
#                     await manager.send_personal_message({"type": "error", "message": f"Failed to ZIP folder {site_path}"}, websocket)

#             overall_results["per_url_results"][url_input] = crawl_summary
        
#         await manager.send_personal_message({"type": "crawl_complete", "status": "success", "results": overall_results}, websocket)
        
#     except Exception as e:
#         logger.critical(f"Overall crawl process failed: {e}", exc_info=True)
#         await manager.send_personal_message({"type": "crawl_complete", "status": "error", "message": f"Crawl process failed: {e}"}, websocket)
#     finally:
#         _is_processing = False
#         logger.info("Crawl process finished. Worker is now idle.")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_json()
#             action = data.get("action")
#             payload = data.get("payload", {})

#             if action == "start_crawl":
#                 if _is_processing:
#                     await manager.send_personal_message({"type": "error", "message": "A crawl is already in progress."}, websocket)
#                     continue
#                 try:
#                     crawl_payload = StartCrawlPayload(**payload)
#                     await manager.send_personal_message({"type": "ack", "message": "Crawl request received and validated. Starting process."}, websocket)
#                     # Run the crawl in the background so it doesn't block the websocket
#                     asyncio.create_task(run_crawl_and_notify(websocket, crawl_payload))
#                 except ValidationError as e:
#                     await manager.send_personal_message({"type": "error", "message": "Invalid payload for start_crawl.", "details": e.errors()}, websocket)

#             elif action == "get_status":
#                 metrics = get_local_machine_metrics()
#                 await manager.send_personal_message({"type": "status_response", "payload": metrics}, websocket)
            
#             else:
#                 await manager.send_personal_message({"type": "error", "message": f"Unknown action: {action}"}, websocket)
    
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#     except Exception as e:
#         logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)
#         # Try to send an error to the client before closing
#         try:
#             await websocket.send_json({"type": "error", "message": "A server-side error occurred."})
#         except:
#             pass # The connection might already be closed
#         manager.disconnect(websocket)

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting FastAPI application with WebSocket support...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import shutil
import asyncio
import os
import csv
import re
import json  # Added json import
from urllib.parse import urljoin, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from typing import List, Dict, Any, Tuple, Optional, Set
import io
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
import platform
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import logging
import sys
from pathlib import Path
from time import time
from urllib.parse import unquote

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Crawling Worker WebSocket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force UTF-8 encoding
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Constants
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}
EXCLUDE_KEYWORDS = ['pdf', 'jpeg', 'jpg', 'png', 'webp', 'login', 'signup']
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Desktop/crawled_data")

# Pydantic Models for WebSocket Message Payloads
class StartCrawlPayload(BaseModel):
    urls: List[str]
    output_dir: str
    max_concurrency: int = 8
    max_depth: int = 2

# Sitemap Crawler Import
try:
    from sitemap_crawler import get_sitemap_data_for_single_url
    SITEMAP_CRAWLER_AVAILABLE = True
    logger.info("Successfully imported sitemap_crawler module.")
except ImportError:
    SITEMAP_CRAWLER_AVAILABLE = False
    logger.warning("sitemap_crawler.py not found. Sitemap processing will be skipped.")

    async def get_sitemap_data_for_single_url(url: str, session: aiohttp.ClientSession) -> List[Tuple[str, str]]:
        logger.error("Sitemap crawler module not available. Returning empty list.")
        return []

# Enhanced URL Validation
def validate_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate and correct a URL. Returns (corrected_url, error_message)."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None, "Invalid URL: missing scheme or netloc"
        
        decoded_url = unquote(url)
        parsed = urlparse(decoded_url)
        
        path_segments = parsed.path.strip('/').split('/')
        corrected_segments = []
        seen_html = False
        for segment in path_segments:
            if segment.endswith('.html'):
                if seen_html:
                    logger.info(f"Removing redundant .html segment: {segment}")
                    continue
                seen_html = True
            corrected_segments.append(segment)
        
        corrected_path = '/' + '/'.join(corrected_segments) if corrected_segments else ''
        corrected_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            corrected_path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        if any(keyword in corrected_url.lower() for keyword in EXCLUDE_KEYWORDS):
            return None, f"URL contains excluded keyword: {corrected_url}"
        
        if corrected_url != url:
            logger.info(f"Corrected URL from {url} to {corrected_url}")
        
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', corrected_url):
            return None, "Invalid URL format after correction"
        
        return corrected_url, None
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return None, str(e)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_states: Dict[WebSocket, bool] = {}
        self.connection_active: Dict[WebSocket, bool] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_states[websocket] = False
        self.connection_active[websocket] = False
        logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.connection_states.pop(websocket, None)
        self.connection_active.pop(websocket, None)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send message to a client: {e}")

manager = ConnectionManager()

# URL Utilities & Core Logic
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
            "crawling_status": "in_use" if any(manager.connection_states.values()) else "idle"
        }
    except Exception as e:
        logger.error(f"Error getting local machine metrics: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def prepare_initial_url_scheme(url_str: str) -> str:
    if not url_str:
        return ""
    url_str = url_str.strip()
    parsed = urlparse(url_str)
    if not parsed.scheme:
        return f"http://{url_str.lstrip('//')}"
    return url_str

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
    """
    Enhanced Markdown cleaning to produce concise, consistent output.
    - Removes UI artifacts (e.g., 'Copy' buttons).
    - Condenses repetitive code examples.
    - Simplifies constructor descriptions into a table.
    - Normalizes parameter formatting.
    - Removes excessive whitespace.
    """
    md_text = re.sub(r'Copy\s*```', '```', md_text, flags=re.IGNORECASE)
    md_text = re.sub(r'\s*Copy\s*\n', '', md_text, flags=re.IGNORECASE)
    md_text = re.sub(r'\n[ \t]*\n+', '\n\n', md_text)
    md_text = md_text.strip()
    md_text = re.sub(r'_([^_\n]+?)_', r'\1', md_text)

    def condense_code_blocks(match):
        code_content = match.group(2).strip()
        lines = code_content.split('\n')
        if 'hashlib' in code_content and 'update' in code_content:
            if any('m.update(' in line for line in lines) and 'm.hexdigest()' in code_content:
                data_parts = []
                for line in lines:
                    if 'm.update(b' in line:
                        part = re.search(r'm\.update\((b".*?")\)', line)
                        if part:
                            data_parts.append(part.group(1).strip('b"').strip('"'))
                if data_parts:
                    combined_data = ''.join(data_parts)
                    hexdigest_line = next((line for line in lines if '.hexdigest()' in line and '031edd7d' in line), None)
                    if hexdigest_line:
                        return f'```python\n>>> hashlib.sha256(b"{combined_data}").hexdigest()\n{hexdigest_line.split()[-1]}\n```'
        return match.group(0)

    md_text = re.sub(r'```(python)?\n(.*?)\n```', condense_code_blocks, md_text, flags=re.DOTALL)

    constructor_pattern = r'(hashlib\.\w+\(\[data, \]\*, usedforsecurity=True\))¶\n+'
    constructors = re.findall(constructor_pattern, md_text)
    if len(constructors) > 1:
        table = "| Constructor | Description |\n|-------------|-------------|\n"
        for constructor in sorted(set(constructors)):
            name = constructor.split('(')[0]
            table += f"| `{name}` | Hash algorithm constructor. Accepts optional `data` and `usedforsecurity=True`. |\n"
        md_text = re.sub(
            r'(## Constructors¶\n)((?:hashlib\.\w+\(\[data, \]\*, usedforsecurity=True\)¶\n+)+)',
            r'\1\n' + table + '\n',
            md_text
        )

    md_text = re.sub(
        r'More condensed:\s*```python\n>>> hashlib\.sha256\(b".*?"\)\.hexdigest\(\)\n[\'"].*?[\'"]\n```\n',
        '',
        md_text,
        flags=re.DOTALL
    )

    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    return md_text.strip()

def sanitize_filename(url: str) -> str:
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.replace(".", "")
        path = parsed.path.strip("/").replace("/", "").replace(".", "") if parsed.path else "index"
        query_safe = re.sub(r'[^a-zA-Z0-9_-]', '', parsed.query[:50]) if parsed.query else ""
        filename_base = f"{netloc}{path}{query_safe}".strip()
        filename_base = re.sub(r'[-_]+', '-', filename_base)
        return filename_base[:250] + ".md"
    except Exception:
        return f"url{abs(hash(url))}.md"

def sanitize_dirname(url: str) -> str:
    try:
        return re.sub(r'[^a-zA-Z0-9_-]', '', urlparse(url).netloc)
    except Exception:
        return f"domain{abs(hash(url))}"

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
    max_depth: int,
    websocket: Optional[WebSocket] = None
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "success": [], "failed": [], "skipped_by_filter": [],
        "initial_url": start_url_original_schemed,
        "effective_start_url": None,
        "output_path_for_site": None
    }

    async def send_progress(message: str, detail: Optional[Dict] = None, throttle_interval: float = 1.0):
        if not hasattr(send_progress, 'last_sent'):
            send_progress.last_sent = 0
        current_time = time()
        if websocket and (current_time - send_progress.last_sent >= throttle_interval):
            payload = {"type": "crawl_progress", "message": message, "detail": detail or {}}
            await manager.send_personal_message(payload, websocket)
            send_progress.last_sent = current_time

    validated_url, validation_error = validate_url(start_url_original_schemed)
    if validation_error:
        results["failed"].append({"url": start_url_original_schemed, "error": f"Validation failed: {validation_error}"})
        await send_progress(f"Invalid URL: {start_url_original_schemed} ({validation_error})")
        return results
    start_url_original_schemed = validated_url

    await send_progress(f"Starting crawl for URL: {start_url_original_schemed}")
    async with aiohttp.ClientSession(headers=COMMON_HEADERS) as http_session:
        effective_start_url, error_msg = await resolve_initial_url(http_session, start_url_original_schemed)
        if not effective_start_url:
            results["failed"].append({"url": start_url_original_schemed, "error": f"Resolution failed: {error_msg}"})
            await send_progress(f"URL resolution failed: {error_msg}")
            return results

    results["effective_start_url"] = effective_start_url
    await send_progress(f"URL resolved to: {effective_start_url}")

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
        await send_progress(f"Output directory set to: {site_output_path_specific}")

    except Exception as e:
        results["failed"].append({"url": start_url_original_schemed, "error": f"Directory setup failed: {e}"})
        logger.error(f"Directory setup failed for {start_url_original_schemed}: {e}", exc_info=True)
        await send_progress(f"Error setting up directory: {e}")
        return results

    md_generator = DefaultMarkdownGenerator(options={"ignore_links": True, "escape_html": True, "body_width": 0})
    config = CrawlerRunConfig(markdown_generator=md_generator, cache_mode="BYPASS", exclude_social_media_links=True)

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
                    if crawl_queue.empty():
                        break
                    continue
                except asyncio.CancelledError:
                    break
                
                if current_url in crawled_urls:
                    crawl_queue.task_done()
                    continue
                
                crawled_urls.add(current_url)
                progress_msg = f"Crawling ({len(crawled_urls)}/{len(queued_urls)}): {current_url} (Depth: {current_depth})"
                logger.info(progress_msg)
                await send_progress(progress_msg, {"crawled_count": len(crawled_urls), "queued_count": len(queued_urls)})
                
                try:
                    async with semaphore:
                        async with AsyncWebCrawler(verbose=False) as crawler:
                            page_data = await crawler.arun(url=current_url, config=config)

                    final_url_processed = normalize_url_for_deduplication(page_data.url or current_url)

                    if not page_data.success:
                        logger.warning(f"Crawl failed for {current_url}: {page_data.error_message}")
                        results["failed"].append({"url": current_url, "error": page_data.error_message})
                        await send_progress(f"Crawl failed for {current_url}", {"error": page_data.error_message})
                        crawl_queue.task_done()
                        continue

                    if hasattr(page_data, 'status_code') and page_data.status_code == 404:
                        logger.warning(f"404 Not Found for {current_url}. Skipping Markdown generation.")
                        results["failed"].append({"url": current_url, "error": "HTTP 404 Not Found"})
                        await send_progress(f"Skipped {current_url}: HTTP 404 Not Found")
                        crawl_queue.task_done()
                        continue

                    if page_data.markdown and "404" in page_data.markdown.raw_markdown.lower():
                        logger.warning(f"404 error content detected in {current_url}. Skipping Markdown generation.")
                        results["failed"].append({"url": current_url, "error": "404 error content detected"})
                        await send_progress(f"Skipped {current_url}: 404 error content detected")
                        crawl_queue.task_done()
                        continue

                    if page_data.markdown:
                        filename = sanitize_filename(final_url_processed)
                        output_path = os.path.join(current_path, filename)
                        loop = asyncio.get_running_loop()
                        process_result = await loop.run_in_executor(
                            executor, process_markdown_and_save, final_url_processed, page_data.markdown.raw_markdown, output_path)
                        
                        if process_result["status"] == "success":
                            results["success"].append(final_url_processed)
                            await send_progress(f"Saved: {final_url_processed}", {"path": output_path})
                        else:
                            results["failed"].append({"url": final_url_processed, "error": process_result["error"]})
                            await send_progress(f"Save failed for {final_url_processed}", {"error": process_result["error"]})
                    else:
                        logger.warning(f"No markdown content for {final_url_processed}")
                        results["failed"].append({"url": final_url_processed, "error": "No markdown content"})
                        await send_progress(f"No markdown content for {final_url_processed}")

                    if current_depth < max_depth and page_data.links:
                        all_links = page_data.links.get("internal", []) + page_data.links.get("external", [])
                        for link_info in all_links:
                            href = link_info.get("href")
                            if not href:
                                continue
                            try:
                                absolute_url = urljoin(final_url_processed, href)
                                validated_link, link_error = validate_url(absolute_url)
                                if link_error:
                                    logger.debug(f"Skipping invalid link {absolute_url}: {link_error}")
                                    results["skipped_by_filter"].append({"url": absolute_url, "reason": link_error})
                                    continue
                                normalized_url = normalize_url_for_deduplication(validated_link)
                                if urlparse(normalized_url).netloc == expected_domain:
                                    if normalized_url not in crawled_urls and normalized_url not in queued_urls:
                                        queued_urls.add(normalized_url)
                                        await crawl_queue.put((normalized_url, current_depth + 1, expected_domain, current_path))
                            except Exception as link_e:
                                logger.warning(f"Link processing error for '{href}': {link_e}")
                
                except Exception as e_crawl:
                    logger.error(f"Critical worker error for {current_url}: {e_crawl}", exc_info=True)
                    results["failed"].append({"url": current_url, "error": f"Worker exception: {e_crawl}"})
                    await send_progress(f"Critical error for {current_url}", {"error": str(e_crawl)})
                finally:
                    crawl_queue.task_done()

        worker_tasks = [asyncio.create_task(crawl_page_worker()) for _ in range(max_concurrency)]
        await crawl_queue.join()
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    logger.info(f"Finished crawl for {start_url_original_schemed}")
    await send_progress(f"Finished crawl for {start_url_original_schemed}")
    return results

async def process_and_save_sitemap(effective_url: str, output_path: str, websocket: Optional[WebSocket] = None) -> Dict[str, Any]:
    sitemap_results = {"status": "initiated", "based_on_effective_url": effective_url}

    async def send_sitemap_progress(message: str, detail: Optional[Dict] = None, throttle_interval: float = 1.0):
        if not hasattr(send_sitemap_progress, 'last_sent'):
            send_sitemap_progress.last_sent = 0
        current_time = time()
        if websocket and (current_time - send_sitemap_progress.last_sent >= throttle_interval):
            payload = {"type": "sitemap_progress", "message": message, "detail": detail or {}}
            await manager.send_personal_message(payload, websocket)
            send_sitemap_progress.last_sent = current_time
    
    await send_sitemap_progress("Sitemap processing started.")
    if not SITEMAP_CRAWLER_AVAILABLE:
        error_msg = "Sitemap crawler module not available."
        sitemap_results.update({"status": "skipped", "error": error_msg})
        await send_sitemap_progress(f"Skipped: {error_msg}")
        return sitemap_results

    try:
        async with aiohttp.ClientSession(headers=COMMON_HEADERS) as session:
            sitemap_entries = await get_sitemap_data_for_single_url(effective_url, session)
        
        # Validate sitemap URLs
        valid_entries = []
        for url, lastmod in sitemap_entries:
            validated_url, error = validate_url(url)
            if error:
                logger.debug(f"Skipping invalid sitemap URL {url}: {error}")
                continue
            valid_entries.append((validated_url, lastmod))
        
        sitemap_results["total_sitemap_entries_returned_by_crawler_module"] = len(sitemap_entries)
        sitemap_results["valid_sitemap_entries"] = len(valid_entries)
        if not valid_entries:
            error_msg = "No valid URLs with lastmod found in sitemap."
            sitemap_results.update({"status": "no_valid_sitemap_data_found_by_module", "error": error_msg})
            await send_sitemap_progress(f"No valid sitemap data found.")
            return sitemap_results

        # Save as CSV
        csv_path = Path(output_path) / "sitemap_data.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "last_modified"])
            writer.writerows(valid_entries)
        logger.info(f"Saved sitemap data to CSV: {csv_path}")
        
        sitemap_results.update({
            "status": "success",
            "sitemap_csv_path": str(csv_path),
            "sitemap_urls_found_with_valid_lastmod": len(valid_entries)
        })
        await send_sitemap_progress(
            f"Successfully saved {len(valid_entries)} sitemap URLs to CSV.",
            {"csv_path": str(csv_path)}
        )
    except Exception as e:
        logger.error(f"Sitemap processing error for {effective_url}: {e}", exc_info=True)
        sitemap_results.update({"status": "error", "error": str(e)})
        await send_sitemap_progress("Sitemap processing failed.", {"error": str(e)})
    return sitemap_results

async def run_crawl_and_notify(websocket: WebSocket, payload: StartCrawlPayload):
    try:
        if manager.connection_states[websocket]:
            await manager.send_personal_message({"type": "error", "message": "A crawl is already in progress for this connection."}, websocket)
            return
        manager.connection_states[websocket] = True
        overall_results = {"per_url_results": {}}

        os.makedirs(os.path.abspath(DEFAULT_OUTPUT_DIR), exist_ok=True)
        for url_input in payload.urls:
            await manager.send_personal_message({"type": "info", "message": f"Starting processing for URL: {url_input}"}, websocket)
            schemed_url = prepare_initial_url_scheme(url_input)
            
            crawl_summary = await crawl_website_single_site(
                start_url_original_schemed=schemed_url,
                output_dir=DEFAULT_OUTPUT_DIR,
                max_concurrency=payload.max_concurrency,
                max_depth=payload.max_depth,
                websocket=websocket
            )
            
            site_path = crawl_summary.get("output_path_for_site")
            effective_url = crawl_summary.get("effective_start_url")

            if site_path and effective_url and os.path.isdir(site_path):
                sitemap_summary = await process_and_save_sitemap(effective_url, site_path, websocket)
                crawl_summary["sitemap_processing_results"] = sitemap_summary

                metadata_path = Path(site_path) / "crawl_metadata.json"
                metadata_content = {
                    "crawl_summary_snapshot": crawl_summary,
                    "timestamp": time(),
                    "input_url": url_input,
                    "effective_url": effective_url
                }
                try:
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata_content, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved metadata to: {metadata_path}")
                except Exception as e:
                    logger.error(f"Failed to save metadata to {metadata_path}: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Failed to save metadata for {url_input}: {str(e)}"
                    }, websocket)
                
                zip_base_name = os.path.join(DEFAULT_OUTPUT_DIR, f"{os.path.basename(site_path)}_output")
                zip_file = create_zip_archive(site_path, zip_base_name)
                
                if zip_file:
                    crawl_summary["site_output_zip_file"] = zip_file
                    await manager.send_personal_message({"type": "info", "message": f"Created ZIP archive: {zip_file}"}, websocket)
                    try:
                        shutil.rmtree(site_path)
                        crawl_summary["data_folder_deleted_after_zip"] = True
                    except Exception as e:
                        logger.error(f"Failed to delete folder {site_path}: {e}")
                        crawl_summary["data_folder_deleted_after_zip"] = False
                else:
                    logger.error(f"Failed to zip folder {site_path}")
                    await manager.send_personal_message({"type": "error", "message": f"Failed to ZIP folder {site_path}"}, websocket)

            overall_results["per_url_results"][url_input] = crawl_summary
        
        await manager.send_personal_message({"type": "crawl_complete", "status": "success", "results": overall_results}, websocket)
    
    except Exception as e:
        logger.critical(f"Overall crawl process failed: {e}", exc_info=True)
        await manager.send_personal_message({"type": "crawl_complete", "status": "error", "message": f"Crawl process failed: {e}"}, websocket)
    finally:
        manager.connection_states[websocket] = False
        logger.info("Crawl process finished. Worker is now idle.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        manager.connection_active[websocket] = True
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                action = data.get("action")
                payload = data.get("payload", {})

                if action == "start_crawl":
                    if manager.connection_states[websocket]:
                        await manager.send_personal_message({"type": "error", "message": "A crawl is already in progress."}, websocket)
                        continue
                    try:
                        crawl_payload = StartCrawlPayload(**payload)
                        await manager.send_personal_message({"type": "ack", "message": "Crawl request received and validated. Starting process."}, websocket)
                        asyncio.create_task(run_crawl_and_notify(websocket, crawl_payload))
                    except ValidationError as e:
                        await manager.send_personal_message({"type": "error", "message": "Invalid payload for start_crawl.", "details": e.errors()}, websocket)

                elif action == "get_status":
                    metrics = get_local_machine_metrics()
                    await manager.send_personal_message({"type": "status_response", "payload": metrics}, websocket)
                
                else:
                    await manager.send_personal_message({"type": "error", "message": f"Unknown action: {action}"}, websocket)

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
    except WebSocketDisconnect:
        if manager.connection_active.get(websocket, False):
            logger.info(f"WebSocket client disconnected gracefully")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "A server-side error occurred."})
        except:
            pass
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application with WebSocket support...")
    uvicorn.run(app, host="0.0.0.0", port=8000)