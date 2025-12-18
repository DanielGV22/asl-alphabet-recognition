import time
import psutil
import os
from collections import deque

class PerfMeter:
    def __init__(self, fps_window=30):
        self.times = deque(maxlen=fps_window)
        self.proc = psutil.Process(os.getpid())
        self.last_t = None

    def tick(self):
        now = time.perf_counter()
        if self.last_t is not None:
            self.times.append(now - self.last_t)
        self.last_t = now

    def fps(self):
        if not self.times:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))

    def cpu_percent(self):
        return self.proc.cpu_percent(interval=None)

    def ram_mb(self):
        return self.proc.memory_info().rss / (1024 * 1024)

def gpu_mem_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None
