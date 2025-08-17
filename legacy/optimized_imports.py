"""
Optimized imports for better performance and smaller bundle size
"""

# Lazy imports for better startup time
def get_pandas():
    import pandas as pd
    return pd

def get_numpy():
    import numpy as np
    return np

def get_coinbase_client():
    from coinbase.rest import RESTClient
    return RESTClient

def get_openai_client():
    import openai
    return openai

# Only import what we need from scipy
def get_scipy_stats():
    from scipy import stats
    return stats

# Memory-efficient data structures
class RingBuffer:
    """Memory-efficient circular buffer for price data"""
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self.data = []
        self.index = 0
    
    def append(self, item):
        if len(self.data) < self.maxsize:
            self.data.append(item)
        else:
            self.data[self.index] = item
            self.index = (self.index + 1) % self.maxsize
    
    def get_data(self):
        if len(self.data) < self.maxsize:
            return self.data
        return self.data[self.index:] + self.data[:self.index]

# Cache decorator for expensive functions
def cache_result(duration=300):
    """Cache function results for specified duration"""
    def decorator(func):
        cache = {}
        def wrapper(*args, **kwargs):
            import time
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < duration:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        return wrapper
    return decorator
