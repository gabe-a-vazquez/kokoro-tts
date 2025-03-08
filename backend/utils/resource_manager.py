import psutil
import torch
from typing import Dict

class ResourceManager:
    def __init__(self):
        self.memory_threshold = 95  # Percentage
        self.cpu_threshold = 98     # Percentage

    def get_system_stats(self) -> Dict[str, float]:
        """Get current system resource usage"""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        else:
            stats["gpu_memory_allocated"] = 0
            
        return stats

    def check_resources(self) -> None:
        """Check if system resources are available"""
        stats = self.get_system_stats()
        
        if stats["memory_percent"] > self.memory_threshold:
            raise Exception(f"System memory usage too high: {stats['memory_percent']}%")
            
        if stats["cpu_percent"] > self.cpu_threshold:
            raise Exception(f"CPU usage too high: {stats['cpu_percent']}%")
            
        if stats["gpu_available"] and stats["gpu_memory_allocated"] > self.memory_threshold:
            raise Exception(f"GPU memory usage too high: {stats['gpu_memory_allocated']}%")
