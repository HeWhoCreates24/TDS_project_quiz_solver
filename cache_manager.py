"""
Smart caching system for quiz solver
Provides multi-layer caching with TTL support for files, transcriptions, parsing, and rendering
"""
import time
import hashlib
import threading
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CachedItem:
    """Container for cached content with metadata"""
    content: Any
    timestamp: float
    size: int = 0
    
    def is_expired(self, ttl: Optional[int]) -> bool:
        """Check if cached item has exceeded its TTL"""
        if ttl is None:  # Permanent cache
            return False
        return (time.time() - self.timestamp) > ttl


class QuizCache:
    """Multi-layer cache for quiz solving operations"""
    
    def __init__(self):
        # Cache layers
        self.file_cache: Dict[str, CachedItem] = {}  # URL → downloaded file
        self.transcription_cache: Dict[str, CachedItem] = {}  # audio_hash → transcription
        self.parse_cache: Dict[str, CachedItem] = {}  # content_hash → parsed data
        self.render_cache: Dict[str, CachedItem] = {}  # URL → rendered HTML
        
        # Thread safety
        self._cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved_seconds": 0,
            "file_hits": 0,
            "transcription_hits": 0,
            "parse_hits": 0,
            "render_hits": 0
        }
    
    # ===== File Download Cache =====
    
    def get_file(self, url: str, ttl: int = 3600) -> Optional[Dict[str, Any]]:
        """
        Get cached file download result
        
        Args:
            url: File URL
            ttl: Time to live in seconds (default 1 hour)
        
        Returns:
            Cached file data or None if not found/expired
        """
        with self._cache_lock:
            if url not in self.file_cache:
                return None
            
            cached = self.file_cache[url]
            if cached.is_expired(ttl):
                # Remove expired entry
                del self.file_cache[url]
                logger.debug(f"[CACHE_EXPIRED] File: {url}")
                return None
            
            self.stats["hits"] += 1
            self.stats["file_hits"] += 1
            logger.info(f"[CACHE_HIT] File: {url}")
            return cached.content
    
    def set_file(self, url: str, content: Dict[str, Any], ttl: int = 3600):
        """Cache file download result"""
        with self._cache_lock:
            size = content.get("size", 0)
            self.file_cache[url] = CachedItem(
                content=content,
                timestamp=time.time(),
                size=size
            )
            logger.debug(f"[CACHE_SET] File: {url} ({size} bytes)")
    
    # ===== Transcription Cache =====
    
    def get_transcription(self, audio_hash: str, ttl: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached transcription result
        
        Args:
            audio_hash: SHA256 hash of audio file
            ttl: Time to live (None = permanent)
        
        Returns:
            Cached transcription or None if not found
        """
        with self._cache_lock:
            if audio_hash not in self.transcription_cache:
                return None
            
            cached = self.transcription_cache[audio_hash]
            if cached.is_expired(ttl):
                del self.transcription_cache[audio_hash]
                logger.debug(f"[CACHE_EXPIRED] Transcription: {audio_hash[:8]}...")
                return None
            
            self.stats["hits"] += 1
            self.stats["transcription_hits"] += 1
            logger.info(f"[CACHE_HIT] Transcription: {audio_hash[:8]}...")
            return cached.content
    
    def set_transcription(self, audio_hash: str, content: Dict[str, Any], ttl: Optional[int] = None):
        """Cache transcription result (permanent by default)"""
        with self._cache_lock:
            self.transcription_cache[audio_hash] = CachedItem(
                content=content,
                timestamp=time.time(),
                size=len(str(content))
            )
            logger.debug(f"[CACHE_SET] Transcription: {audio_hash[:8]}...")
    
    # ===== Parse Cache =====
    
    def get_parsed_csv(self, content_hash: str, ttl: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached CSV parse result
        
        Args:
            content_hash: SHA256 hash of CSV content
            ttl: Time to live (None = permanent)
        
        Returns:
            Cached parse result or None if not found
        """
        with self._cache_lock:
            if content_hash not in self.parse_cache:
                return None
            
            cached = self.parse_cache[content_hash]
            if cached.is_expired(ttl):
                del self.parse_cache[content_hash]
                logger.debug(f"[CACHE_EXPIRED] Parse: {content_hash[:8]}...")
                return None
            
            self.stats["hits"] += 1
            self.stats["parse_hits"] += 1
            logger.info(f"[CACHE_HIT] CSV Parse: {content_hash[:8]}...")
            return cached.content
    
    def set_parsed_csv(self, content_hash: str, content: Dict[str, Any], ttl: Optional[int] = None):
        """Cache CSV parse result (permanent by default)"""
        with self._cache_lock:
            self.parse_cache[content_hash] = CachedItem(
                content=content,
                timestamp=time.time(),
                size=len(str(content))
            )
            logger.debug(f"[CACHE_SET] CSV Parse: {content_hash[:8]}...")
    
    # ===== Render Cache =====
    
    def get_rendered_page(self, url: str, ttl: int = 300) -> Optional[Dict[str, Any]]:
        """
        Get cached rendered page
        
        Args:
            url: Page URL
            ttl: Time to live in seconds (default 5 minutes)
        
        Returns:
            Cached rendered page or None if not found/expired
        """
        with self._cache_lock:
            if url not in self.render_cache:
                return None
            
            cached = self.render_cache[url]
            if cached.is_expired(ttl):
                del self.render_cache[url]
                logger.debug(f"[CACHE_EXPIRED] Render: {url}")
                return None
            
            self.stats["hits"] += 1
            self.stats["render_hits"] += 1
            logger.info(f"[CACHE_HIT] Rendered page: {url}")
            return cached.content
    
    def set_rendered_page(self, url: str, content: Dict[str, Any], ttl: int = 300):
        """Cache rendered page result"""
        with self._cache_lock:
            size = len(str(content))
            self.render_cache[url] = CachedItem(
                content=content,
                timestamp=time.time(),
                size=size
            )
            logger.debug(f"[CACHE_SET] Rendered page: {url}")
    
    # ===== Statistics & Utilities =====
    
    def record_miss(self):
        """Record a cache miss"""
        with self._cache_lock:
            self.stats["misses"] += 1
    
    def record_time_saved(self, seconds: float):
        """Record time saved by cache hit"""
        with self._cache_lock:
            self.stats["saved_seconds"] += seconds
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "saved_seconds": self.stats["saved_seconds"],
                "breakdown": {
                    "file_hits": self.stats["file_hits"],
                    "transcription_hits": self.stats["transcription_hits"],
                    "parse_hits": self.stats["parse_hits"],
                    "render_hits": self.stats["render_hits"]
                },
                "cache_sizes": {
                    "files": len(self.file_cache),
                    "transcriptions": len(self.transcription_cache),
                    "parses": len(self.parse_cache),
                    "renders": len(self.render_cache)
                }
            }
    
    def log_stats(self):
        """Log cache statistics"""
        stats = self.get_stats()
        logger.info(
            f"[CACHE_STATS] Hits: {stats['hits']}, Misses: {stats['misses']}, "
            f"Hit Rate: {stats['hit_rate']:.1f}%, Time Saved: {stats['saved_seconds']:.1f}s"
        )
        logger.info(
            f"[CACHE_BREAKDOWN] Files: {stats['breakdown']['file_hits']}, "
            f"Transcriptions: {stats['breakdown']['transcription_hits']}, "
            f"Parses: {stats['breakdown']['parse_hits']}, "
            f"Renders: {stats['breakdown']['render_hits']}"
        )
    
    def clear(self):
        """Clear all caches"""
        with self._cache_lock:
            self.file_cache.clear()
            self.transcription_cache.clear()
            self.parse_cache.clear()
            self.render_cache.clear()
            logger.info("[CACHE_CLEAR] All caches cleared")
    
    def clear_expired(self):
        """Remove all expired cache entries"""
        with self._cache_lock:
            # Check file cache (1 hour TTL)
            expired_files = [
                url for url, cached in self.file_cache.items()
                if cached.is_expired(3600)
            ]
            for url in expired_files:
                del self.file_cache[url]
            
            # Check render cache (5 min TTL)
            expired_renders = [
                url for url, cached in self.render_cache.items()
                if cached.is_expired(300)
            ]
            for url in expired_renders:
                del self.render_cache[url]
            
            if expired_files or expired_renders:
                logger.info(f"[CACHE_CLEANUP] Removed {len(expired_files)} files, {len(expired_renders)} renders")


# ===== Content Hashing Utilities =====

def hash_content(content: bytes) -> str:
    """
    Compute SHA256 hash of content
    
    Args:
        content: Raw bytes to hash
    
    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(content).hexdigest()


def hash_file(file_path: str) -> str:
    """
    Compute SHA256 hash of file
    
    Args:
        file_path: Path to file
    
    Returns:
        Hex digest of SHA256 hash
    """
    try:
        with open(file_path, 'rb') as f:
            return hash_content(f.read())
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        raise


# Global cache instance
quiz_cache = QuizCache()
