import aiohttp
import asyncio
import logging
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
import gzip
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_sitemap_content(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Fetch sitemap content, handling XML and gzipped files."""
    try:
        async with session.get(url, timeout=20) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch sitemap {url}: HTTP {response.status}")
                return None
            content = await response.read()
            if url.endswith('.gz'):
                content = gzip.decompress(content)
            return content
    except Exception as e:
        logger.error(f"Error fetching sitemap {url}: {e}")
        return None

async def parse_sitemap(xml_content: bytes, base_url: str) -> List[Tuple[str, str]]:
    """Parse XML sitemap and extract URLs with lastmod."""
    entries = []
    try:
        root = ET.fromstring(xml_content)
        namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        if root.tag.endswith('sitemapindex'):
            for sitemap in root.findall('s:/sitemap', namespace):
                loc = sitemap.find('s:loc', namespace)
                if loc is not None and loc.text:
                    sub_sitemap_url = loc.text.strip()
                    async with aiohttp.ClientSession() as session:
                        sub_content = await fetch_sitemap_content(session, sub_sitemap_url)
                        if sub_content:
                            sub_entries = await parse_sitemap(sub_content, base_url)
                            entries.extend(sub_entries)
        else:
            for url_elem in root.findall('s:/url', namespace):
                loc = url_elem.find('s:loc', namespace)
                lastmod = url_elem.find('s:lastmod', namespace)
                if loc is not None and loc.text and lastmod is not None and lastmod.text:
                    url = urljoin(base_url, loc.text.strip())
                    entries.append((url, lastmod.text.strip()))
    except ET.ParseError:
        logger.warning(f"Invalid XML in sitemap for {base_url}")
    except Exception as e:
        logger.error(f"Error parsing sitemap for {base_url}: {e}")
    return entries

async def get_sitemap_data_for_single_url(url: str, session: aiohttp.ClientSession) -> List[Tuple[str, str]]:
    """Fetch and parse sitemap for a given URL, returning (url, lastmod) tuples."""
    parsed = urlparse(url)
    sitemap_urls = [
        f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap.xml.gz"
    ]
    
    entries = []
    for sitemap_url in sitemap_urls:
        logger.info(f"Trying sitemap: {sitemap_url}")
        content = await fetch_sitemap_content(session, sitemap_url)
        if content:
            new_entries = await parse_sitemap(content, url)
            entries.extend(new_entries)
            break
    
    seen_urls = set()
    valid_entries = []
    for entry_url, lastmod in entries:
        if entry_url not in seen_urls:
            seen_urls.add(entry_url)
            valid_entries.append((entry_url, lastmod))
    
    logger.info(f"Found {len(valid_entries)} valid sitemap entries for {url}")
    return valid_entries