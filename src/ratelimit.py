import time
import logging
import asyncio
from typing import Dict, Optional

logger = logging.getLogger("ratelimit")

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self):
        # Default rate limits per model
        self.rate_limits = {
            "gemini-2.5-pro": {
                "min_interval": 12,  # seconds between requests
                "retries": 1,        # number of retries on rate limit
                "backoff": 60        # backoff time in seconds
            },
            "default": {
                "min_interval": 2,   # default interval for other models
                "retries": 1,
                "backoff": 30
            }
        }
        
        # Track last request times
        self._last_requests: Dict[str, float] = {}
    
    def add_rate_limit(self, model: str, min_interval: int, retries: int = 1, backoff: int = 60):
        """Add or update rate limit for a model"""
        self.rate_limits[model] = {
            "min_interval": min_interval,
            "retries": retries,
            "backoff": backoff
        }
        logger.info(f"Added rate limit for {model}: {min_interval}s interval, {retries} retries")
    
    def get_rate_limit(self, model: str) -> Dict:
        """Get rate limit settings for a model"""
        return self.rate_limits.get(model, self.rate_limits["default"])
    
    async def wait_if_needed(self, model: str):
        """Wait if needed to respect rate limits"""
        settings = self.get_rate_limit(model)
        min_interval = settings["min_interval"]
        
        last_request = self._last_requests.get(model, 0)
        current_time = time.time()
        
        wait_time = min_interval - (current_time - last_request)
        if wait_time > 0:
            logger.debug(f"Rate limit: Waiting {wait_time:.2f}s for {model}")
            await asyncio.sleep(wait_time)
        
        self._last_requests[model] = time.time()
    
    async def handle_rate_limit_error(self, model: str, error: Exception) -> Optional[int]:
        """Handle rate limit error, returns delay time if should retry"""
        settings = self.get_rate_limit(model)
        
        if "RESOURCE_EXHAUSTED" in str(error):
            # Try to extract delay from error message
            import re
            match = re.search(r"'retryDelay': '(\d+)s'", str(error))
            delay = int(match.group(1)) if match else settings["backoff"]
            
            logger.warning(f"Rate limit hit for {model}, retry delay: {delay}s")
            return delay
            
        return None
    
    def record_request(self, model: str):
        """Record a successful request"""
        self._last_requests[model] = time.time()

# Global instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get singleton RateLimiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
