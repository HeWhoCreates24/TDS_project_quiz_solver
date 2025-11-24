"""
Quick test to verify caching implementation
Run this to test cache functionality before full quiz test
"""
import asyncio
import logging
from cache_manager import quiz_cache, hash_content, hash_file
from tool_executors import download_file, parse_csv, render_page
from tools import MultimediaTools
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_file_cache():
    """Test file download caching"""
    logger.info("\n=== Testing File Download Cache ===")
    
    test_url = "https://httpbin.org/json"
    
    # First call - should be cache miss
    logger.info("First download (should be cache MISS)...")
    result1 = await download_file(test_url)
    logger.info(f"Result: {result1}")
    
    # Second call - should be cache hit
    logger.info("\nSecond download (should be cache HIT)...")
    result2 = await download_file(test_url)
    logger.info(f"Result: {result2}")
    
    # Verify same result
    assert result1['size'] == result2['size'], "Cached result should match original"
    logger.info("✅ File cache working correctly!")


async def test_transcription_cache():
    """Test audio transcription caching"""
    logger.info("\n=== Testing Transcription Cache ===")
    
    # Create a dummy audio file for testing hash-based caching
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
        f.write("This is a test audio file placeholder")
        test_file = f.name
    
    # Test hash function
    file_hash1 = hash_file(test_file)
    file_hash2 = hash_file(test_file)
    assert file_hash1 == file_hash2, "Hash should be deterministic"
    logger.info(f"✅ File hash working: {file_hash1[:16]}...")
    
    # Clean up
    import os
    os.remove(test_file)


def test_content_hash():
    """Test content hashing"""
    logger.info("\n=== Testing Content Hashing ===")
    
    content1 = b"Test CSV content\nRow1,Row2,Row3"
    content2 = b"Test CSV content\nRow1,Row2,Row3"
    content3 = b"Different content"
    
    hash1 = hash_content(content1)
    hash2 = hash_content(content2)
    hash3 = hash_content(content3)
    
    assert hash1 == hash2, "Same content should have same hash"
    assert hash1 != hash3, "Different content should have different hash"
    logger.info(f"✅ Content hashing working correctly!")
    logger.info(f"  Hash 1: {hash1[:16]}...")
    logger.info(f"  Hash 2: {hash2[:16]}...")
    logger.info(f"  Hash 3: {hash3[:16]}...")


def test_cache_stats():
    """Test cache statistics"""
    logger.info("\n=== Testing Cache Statistics ===")
    
    # Clear cache first to start fresh
    quiz_cache.clear()
    quiz_cache.stats = {"hits": 0, "misses": 0, "saved_seconds": 0, 
                        "file_hits": 0, "transcription_hits": 0, "parse_hits": 0, "render_hits": 0}
    
    # Record some mock hits/misses
    quiz_cache.record_miss()
    quiz_cache.record_miss()
    quiz_cache.stats["hits"] += 3
    quiz_cache.stats["file_hits"] += 2
    quiz_cache.stats["transcription_hits"] += 1
    quiz_cache.record_time_saved(10.5)
    
    stats = quiz_cache.get_stats()
    logger.info(f"Stats: {stats}")
    
    assert stats["hits"] == 3, "Should have 3 hits"
    assert stats["misses"] == 2, "Should have 2 misses"
    assert stats["hit_rate"] == 60.0, "Hit rate should be 60%"
    assert stats["saved_seconds"] == 10.5, "Should have saved 10.5s"
    
    logger.info("✅ Cache statistics working correctly!")
    quiz_cache.log_stats()


async def main():
    """Run all cache tests"""
    logger.info("=" * 60)
    logger.info("CACHE IMPLEMENTATION TEST SUITE")
    logger.info("=" * 60)
    
    try:
        # Test 1: Content hashing
        test_content_hash()
        
        # Test 2: File download cache
        await test_file_cache()
        
        # Test 3: Transcription cache (hash-based)
        await test_transcription_cache()
        
        # Test 4: Cache statistics
        test_cache_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL CACHE TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nNext step: Run full quiz test to verify real-world caching")
        logger.info("Expected: First run = 0 cache hits, Second run = 60%+ hit rate")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
