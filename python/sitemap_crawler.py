import aiohttp
import asyncio
import logging
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from aiohttp import ClientSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_content(session: ClientSession, url: str) -> Optional[str]:
    """Fetch content from a URL with a timeout."""
    try:
        async with session.get(url, timeout=5) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                return None
            return await response.text()
    except aiohttp.ClientError as e:
        logger.warning(f"Network error fetching {url}: {e}")
        return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None

async def fetch_robots_txt(session: ClientSession, base_url: str) -> List[str]:
    """Fetch sitemap URLs from robots.txt."""
    parsed = urlparse(base_url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    robots_url = f"{scheme}://{netloc}/robots.txt"
    potential_sitemaps = []

    logger.info(f"Attempting to fetch robots.txt: {robots_url}")
    content = await fetch_content(session, robots_url)
    if content:
        for line in content.splitlines():
            if line.strip().lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                sitemap_abs = urljoin(f"{scheme}://{netloc}/", sitemap_url)
                potential_sitemaps.append(sitemap_abs)
        logger.info(f"Sitemaps found in robots.txt: {potential_sitemaps}")
    else:
        logger.info(f"robots.txt not found or inaccessible for {base_url}")
    
    return potential_sitemaps

async def check_common_sitemap_paths(session: ClientSession, base_url: str) -> List[str]:
    """Check common sitemap paths if robots.txt yields no sitemaps."""
    parsed = urlparse(base_url)
    scheme = parsed.scheme
    netloc = parsed.netloc
    common_paths = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap_news.xml", "/sitemap_pages.xml"]
    potential_sitemaps = []

    for path in common_paths:
        full_url = f"{scheme}://{netloc}{path}"
        try:
            async with session.head(full_url, timeout=5) as response:
                if response.status == 200:
                    potential_sitemaps.append(full_url)
                    logger.info(f"Found common sitemap: {full_url}")
        except aiohttp.ClientError:
            pass
        except Exception as e:
            logger.warning(f"Error checking common sitemap {full_url}: {e}")
    
    return potential_sitemaps

async def parse_sitemap(session: ClientSession, xml_content: str, base_url: str) -> List[Tuple[str, str]]:
    """Parse XML sitemap and extract URLs with lastmod dates."""
    entries = []
    try:
        root = ET.fromstring(xml_content)
        ns = ''
        if '}' in root.tag:
            ns = root.tag.split('}', 1)[0] + '}'
        
        root_tag_local = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        loc_tag = ns + 'loc' if ns else 'loc'
        lastmod_tag = ns + 'lastmod' if ns else 'lastmod'

        if root_tag_local == 'sitemapindex':
            sitemap_tag = ns + 'sitemap' if ns else 'sitemap'
            logger.info(f"Found sitemapindex: {base_url}")
            for child in root.findall(sitemap_tag):
                loc_elem = child.find(loc_tag)
                if loc_elem is not None and loc_elem.text:
                    logger.info(f"Processing nested sitemap: {loc_elem.text}")
                    sub_content = await fetch_content(session, loc_elem.text)
                    if sub_content:
                        sub_entries = await parse_sitemap(session, sub_content, base_url)
                        entries.extend(sub_entries)
        elif root_tag_local == 'urlset':
            logger.info(f"Found urlset: {base_url}")
            for url_elem in root.findall(ns + 'url' if ns else 'url'):
                loc_elem = url_elem.find(loc_tag)
                lastmod_elem = url_elem.find(lastmod_tag)
                if loc_elem is not None and loc_elem.text and lastmod_elem is not None and lastmod_elem.text:
                    url = urljoin(base_url, loc_elem.text.strip())
                    entries.append((url, lastmod_elem.text.strip()))
        else:
            logger.warning(f"Unknown sitemap root tag: {root.tag} for {base_url}")
    except ET.ParseError:
        logger.warning(f"Invalid XML in sitemap for {base_url}")
    except Exception as e:
        logger.error(f"Error parsing sitemap for {base_url}: {e}")
    return entries

async def get_sitemap_data_for_single_url(url: str, session: ClientSession) -> List[Tuple[str, str]]:
    """Fetch and parse sitemaps for a single URL, returning (url, lastmod) tuples."""
    potential_sitemaps = await fetch_robots_txt(session, url)
    
    if not potential_sitemaps:
        logger.info("No sitemaps found in robots.txt. Checking common paths.")
        potential_sitemaps = await check_common_sitemap_paths(session, url)
    
    if not potential_sitemaps:
        parsed = urlparse(url)
        default_sitemap = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
        logger.info(f"No sitemaps found. Trying default: {default_sitemap}")
        potential_sitemaps = [default_sitemap]
    
    all_entries = []
    processed_sitemaps = set()
    
    for sitemap_url in potential_sitemaps:
        if sitemap_url not in processed_sitemaps:
            logger.info(f"Processing sitemap: {sitemap_url}")
            content = await fetch_content(session, sitemap_url)
            if content:
                entries = await parse_sitemap(session, content, url)
                all_entries.extend(entries)
            processed_sitemaps.add(sitemap_url)
    
    # Deduplicate and filter entries with lastmod
    seen_urls = set()
    valid_entries = []
    for entry_url, lastmod in all_entries:
        if entry_url not in seen_urls and lastmod:
            seen_urls.add(entry_url)
            valid_entries.append((entry_url, lastmod))
    
    logger.info(f"Found {len(valid_entries)} valid sitemap entries for {url}")
    return valid_entries