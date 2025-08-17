#!/usr/bin/env python3
"""
Real-time performance monitor for ScalperBot
"""
import time
import psutil
import json
from datetime import datetime
import os

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
    
    def collect_metrics(self):
        """Collect system metrics"""
        process = psutil.Process()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'uptime_seconds': time.time() - self.start_time,
            'open_files': len(process.open_files()),
            'threads': process.num_threads()
        }
        
        self.metrics.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
        
        return metrics
    
    def save_metrics(self):
        """Save metrics to file"""
        with open('logs/performance_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self):
        """Print performance summary"""
        if not self.metrics:
            return
            
        latest = self.metrics[-1]
        print(f"💾 Memory: {latest['memory_mb']:.1f} MB")
        print(f"🖥️  CPU: {latest['cpu_percent']:.1f}%")
        print(f"⏱️  Uptime: {latest['uptime_seconds']:.0f}s")
        print(f"📁 Files: {latest['open_files']}")
        print(f"🧵 Threads: {latest['threads']}")

def monitor_loop():
    """Main monitoring loop"""
    monitor = PerformanceMonitor()
    
    try:
        while True:
            metrics = monitor.collect_metrics()
            monitor.print_summary()
            monitor.save_metrics()
            time.sleep(30)  # Monitor every 30 seconds
    except KeyboardInterrupt:
        print("\n📊 Monitoring stopped")

if __name__ == "__main__":
    monitor_loop()
