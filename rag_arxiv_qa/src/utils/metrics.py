# Metrics collection for monitoring system performance
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import time
from threading import Lock

from rag_arxiv_qa.src.utils.logger import log_metric


class MetricsCollector:
    def __init__(self):
        self._lock = Lock()
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)
        self._gauges = {}
        self._start_time = time.time()
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            self._counters[key] += value
            log_metric(f"{metric_name}_total", self._counters[key], tags)
    
    def record_latency(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a latency measurement."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            self._histograms[key].append(duration)
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
            log_metric(f"{metric_name}_duration_seconds", duration, tags)
    
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            self._gauges[key] = value
            log_metric(metric_name, value, tags)
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value."""
        with self._lock:
            key = self._make_key(metric_name, tags)
            return self._counters.get(key, 0)
    
    def get_latency_stats(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        with self._lock:
            key = self._make_key(metric_name, tags)
            values = sorted(self._histograms.get(key, []))
            
            if not values:
                return {}
            
            n = len(values)
            return {
                "count": n,
                "mean": sum(values) / n,
                "min": min(values),
                "max": max(values),
                "p50": values[int(n * 0.5)],
                "p95": values[int(n * 0.95)] if n > 20 else values[-1],
                "p99": values[int(n * 0.99)] if n > 1 else values[-1],
            }
    
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time
            summary = {
                "uptime_seconds": uptime,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }
            
            latency_stats = {}
            for key in self._histograms:
                latency_stats[key] = self.get_latency_stats(key.split("_")[0], None)
            summary["latency_stats"] = latency_stats
            
            return summary
    
    def _make_key(self, metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}{{{tag_str}}}"


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics
