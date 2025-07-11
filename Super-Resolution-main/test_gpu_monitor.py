#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_monitor import SystemMonitor

def test_gpu_monitoring():
    print("Testing GPU monitoring...")
    
    monitor = SystemMonitor()
    
    print("\n1. Testing direct pynvml import...")
    try:
        import pynvml
        print("✓ pynvml imported successfully")
        
        pynvml.nvmlInit()
        print("✓ nvmlInit() successful")
        
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ Found {device_count} GPU(s)")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            print(f"  GPU {i}: {name}")
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"    Memory: {memory_info.used / (1024**2):.0f}MB / {memory_info.total / (1024**2):.0f}MB")
            
    except Exception as e:
        print(f"✗ pynvml test failed: {e}")
    
    print("\n2. Testing system monitor get_gpu_info()...")
    try:
        gpu_info = monitor.get_gpu_info()
        print(f"GPU info result: {gpu_info}")
        
        if gpu_info.get('nvidia_gpus'):
            print("✓ NVIDIA GPUs detected")
            for i, gpu in enumerate(gpu_info['nvidia_gpus']):
                print(f"  GPU {i}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_percent']:.1f}%)")
        else:
            print("✗ No NVIDIA GPUs detected")
            if gpu_info.get('error'):
                print(f"  Error: {gpu_info['error']}")
            if gpu_info.get('message'):
                print(f"  Message: {gpu_info['message']}")
                
    except Exception as e:
        print(f"✗ get_gpu_info() failed: {e}")
    
    print("\n3. Testing full system stats...")
    try:
        stats = monitor.get_system_stats()
        print(f"System stats keys: {list(stats.keys())}")
        
        if 'gpu' in stats:
            gpu_stats = stats['gpu']
            print(f"GPU stats: {gpu_stats}")
        else:
            print("✗ No GPU stats in system stats")
            
    except Exception as e:
        print(f"✗ get_system_stats() failed: {e}")

if __name__ == "__main__":
    test_gpu_monitoring() 