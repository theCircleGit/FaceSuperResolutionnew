import psutil
import time
import threading
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.monitoring_data = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.max_data_points = 100  # Keep last 100 data points
    
    def get_system_stats(self):
        """Get current system statistics"""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # Network information
            network = psutil.net_io_counters()
            
            # GPU information (if available)
            gpu_info = self.get_gpu_info()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else None,
                    'memory_usage_mb': memory.used / (1024 * 1024),
                    'memory_total_mb': memory.total / (1024 * 1024),
                    'memory_percent': memory.percent
                },
                'disk': {
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                'gpu': gpu_info
            }
            
            return stats
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_gpu_info(self):
        """Get GPU information if available"""
        gpu_info = {}
        
        try:
            # Try to get NVIDIA GPU info using nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info['nvidia_gpus'] = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                # Handle both string and bytes types
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_info['nvidia_gpus'].append({
                    'name': name,
                    'memory_used_mb': memory_info.used / (1024 * 1024),
                    'memory_total_mb': memory_info.total / (1024 * 1024),
                    'memory_percent': (memory_info.used / memory_info.total) * 100,
                    'utilization_percent': utilization.gpu,
                    'temperature_celsius': temperature
                })
                
        except ImportError:
            gpu_info['nvidia_gpus'] = None
            gpu_info['error'] = 'nvidia-ml-py not installed'
            gpu_info['message'] = 'Install with: pip install nvidia-ml-py'
        except Exception as e:
            gpu_info['nvidia_gpus'] = None
            gpu_info['error'] = str(e)
            gpu_info['message'] = 'NVIDIA GPU monitoring failed'
        
        # Try alternative GPU detection methods
        if not gpu_info.get('nvidia_gpus'):
            gpu_info.update(self.get_alternative_gpu_info())
        
        return gpu_info
    
    def get_alternative_gpu_info(self):
        """Get GPU information using alternative methods"""
        alt_gpu_info = {}
        
        try:
            # Try using lspci to detect GPUs
            import subprocess
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or '3D' in line]
                if gpu_lines:
                    alt_gpu_info['detected_gpus'] = gpu_lines
                    alt_gpu_info['message'] = f'Found {len(gpu_lines)} GPU(s) via lspci'
                else:
                    alt_gpu_info['message'] = 'No GPUs detected via lspci'
            else:
                alt_gpu_info['message'] = 'lspci not available'
        except Exception:
            alt_gpu_info['message'] = 'Alternative GPU detection failed'
        
        # Try using /proc/driver/nvidia/gpus if available
        try:
            import os
            if os.path.exists('/proc/driver/nvidia/gpus'):
                gpu_dirs = [d for d in os.listdir('/proc/driver/nvidia/gpus') if d.isdigit()]
                if gpu_dirs:
                    alt_gpu_info['nvidia_driver'] = True
                    alt_gpu_info['message'] = f'NVIDIA driver detected with {len(gpu_dirs)} GPU(s)'
        except Exception:
            pass
        
        return alt_gpu_info
    
    def start_monitoring(self, interval=5):
        """Start continuous monitoring"""
        if self.is_monitoring:
            return False
            
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                stats = self.get_system_stats()
                self.monitoring_data.append(stats)
                
                # Keep only the last max_data_points
                if len(self.monitoring_data) > self.max_data_points:
                    self.monitoring_data = self.monitoring_data[-self.max_data_points:]
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def get_current_stats(self):
        """Get current system stats without storing"""
        return self.get_system_stats()
    
    def get_monitoring_history(self, limit=None):
        """Get monitoring history"""
        if limit:
            return self.monitoring_data[-limit:]
        return self.monitoring_data
    
    def clear_history(self):
        """Clear monitoring history"""
        self.monitoring_data = []

# Global monitor instance
system_monitor = SystemMonitor() 