#!/usr/bin/env python3
"""Monitor Training Performance - GPU Utilization, Memory, and Throughput.

This script monitors training performance metrics in real-time:
    - GPU utilization percentage
    - GPU memory usage
    - Training throughput (samples/sec)
    - CPU usage
    - RAM usage

Usage:
    # Start monitoring in a terminal:
    python scripts/monitor_training_performance.py

    # Start training in Google Colab (run the pipeline notebook)

The script will create a live performance report showing:
    - Real-time GPU utilization
    - Memory consumption trends
    - Training speed metrics
    - Bottleneck detection
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with: pip install psutil")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not installed. Install with: pip install gputil")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("Warning: pynvml not installed. Install with: pip install pynvml")


class PerformanceMonitor:
    """Monitor training performance metrics."""
    
    def __init__(self, gpu_id: int = 0, interval: float = 1.0):
        self.gpu_id = gpu_id
        self.interval = interval
        self.start_time = time.time()
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode('utf-8')
            except:
                self.gpu_handle = None
                self.gpu_name = "Unknown GPU"
        else:
            self.gpu_handle = None
            self.gpu_name = "Unknown GPU"
        
        # Metrics storage
        self.metrics_history = {
            'timestamp': [],
            'gpu_util': [],
            'gpu_mem_used': [],
            'gpu_mem_total': [],
            'cpu_percent': [],
            'ram_percent': [],
        }
    
    def get_gpu_metrics(self):
        """Get current GPU metrics."""
        metrics = {
            'gpu_util': 0.0,
            'gpu_mem_used': 0,
            'gpu_mem_total': 0,
            'gpu_mem_percent': 0.0,
            'gpu_temp': 0.0,
            'gpu_power': 0.0,
        }
        
        if NVML_AVAILABLE and self.gpu_handle:
            try:
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics['gpu_util'] = util.gpu
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                metrics['gpu_mem_used'] = mem_info.used // (1024 ** 2)  # MB
                metrics['gpu_mem_total'] = mem_info.total // (1024 ** 2)  # MB
                metrics['gpu_mem_percent'] = (mem_info.used / mem_info.total) * 100
                
                # Temperature
                metrics['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # W
                metrics['gpu_power'] = power
                
            except Exception as e:
                print(f"Error reading GPU metrics: {e}")
        
        elif GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > self.gpu_id:
                    gpu = gpus[self.gpu_id]
                    metrics['gpu_util'] = gpu.load * 100
                    metrics['gpu_mem_used'] = gpu.memoryUsed
                    metrics['gpu_mem_total'] = gpu.memoryTotal
                    metrics['gpu_mem_percent'] = gpu.memoryUtil * 100
                    metrics['gpu_temp'] = gpu.temperature
            except Exception as e:
                print(f"Error reading GPU metrics: {e}")
        
        return metrics
    
    def get_cpu_ram_metrics(self):
        """Get CPU and RAM metrics."""
        metrics = {
            'cpu_percent': 0.0,
            'ram_percent': 0.0,
            'ram_used': 0,
            'ram_total': 0,
        }
        
        if PSUTIL_AVAILABLE:
            try:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                
                mem = psutil.virtual_memory()
                metrics['ram_percent'] = mem.percent
                metrics['ram_used'] = mem.used // (1024 ** 2)  # MB
                metrics['ram_total'] = mem.total // (1024 ** 2)  # MB
            except Exception as e:
                print(f"Error reading CPU/RAM metrics: {e}")
        
        return metrics
    
    def print_metrics(self, gpu_metrics, cpu_metrics):
        """Print metrics in a formatted table."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        elapsed = time.time() - self.start_time
        
        print("="*80)
        print(f"{'TRAINING PERFORMANCE MONITOR':^80}")
        print("="*80)
        print(f"GPU: {self.gpu_name}")
        print(f"Monitoring Time: {elapsed:.1f}s")
        print(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        # GPU Metrics
        print("\n📊 GPU METRICS")
        print("-"*80)
        gpu_util = gpu_metrics['gpu_util']
        gpu_mem_pct = gpu_metrics['gpu_mem_percent']
        
        # GPU utilization bar
        util_bar = self._get_progress_bar(gpu_util, 100)
        util_status = self._get_utilization_status(gpu_util)
        print(f"  GPU Utilization:  {util_bar} {gpu_util:5.1f}% {util_status}")
        
        # GPU memory bar
        mem_bar = self._get_progress_bar(gpu_mem_pct, 100)
        print(f"  GPU Memory:       {mem_bar} {gpu_mem_pct:5.1f}% "
              f"({gpu_metrics['gpu_mem_used']:,}MB / {gpu_metrics['gpu_mem_total']:,}MB)")
        
        if gpu_metrics['gpu_temp'] > 0:
            temp_status = self._get_temp_status(gpu_metrics['gpu_temp'])
            print(f"  GPU Temperature:  {gpu_metrics['gpu_temp']:.1f}°C {temp_status}")
        
        if gpu_metrics['gpu_power'] > 0:
            print(f"  GPU Power:        {gpu_metrics['gpu_power']:.1f}W")
        
        # CPU/RAM Metrics
        print("\n💻 CPU & RAM METRICS")
        print("-"*80)
        
        cpu_bar = self._get_progress_bar(cpu_metrics['cpu_percent'], 100)
        print(f"  CPU Usage:        {cpu_bar} {cpu_metrics['cpu_percent']:5.1f}%")
        
        ram_bar = self._get_progress_bar(cpu_metrics['ram_percent'], 100)
        print(f"  RAM Usage:        {ram_bar} {cpu_metrics['ram_percent']:5.1f}% "
              f"({cpu_metrics['ram_used']:,}MB / {cpu_metrics['ram_total']:,}MB)")
        
        # Analysis
        print("\n🔍 PERFORMANCE ANALYSIS")
        print("-"*80)
        self._print_analysis(gpu_util, gpu_mem_pct, cpu_metrics['cpu_percent'])
        
        print("\n" + "="*80)
        print("Press Ctrl+C to stop monitoring")
        print("="*80)
    
    def _get_progress_bar(self, value, max_value, width=30):
        """Create a progress bar string."""
        filled = int((value / max_value) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def _get_utilization_status(self, util):
        """Get GPU utilization status."""
        if util >= 95:
            return "✅ EXCELLENT"
        elif util >= 80:
            return "✓ GOOD"
        elif util >= 60:
            return "⚠️  MODERATE"
        elif util >= 40:
            return "⚠️  LOW"
        else:
            return "❌ VERY LOW"
    
    def _get_temp_status(self, temp):
        """Get GPU temperature status."""
        if temp >= 85:
            return "🔥 HOT"
        elif temp >= 75:
            return "⚠️  WARM"
        else:
            return "✓ OK"
    
    def _print_analysis(self, gpu_util, gpu_mem, cpu):
        """Print performance analysis."""
        issues = []
        recommendations = []
        
        # Check GPU utilization
        if gpu_util < 40:
            issues.append("⚠️  GPU utilization is LOW (<40%)")
            recommendations.append("• Increase num_workers in DataLoader")
            recommendations.append("• Increase batch_size if VRAM allows")
            recommendations.append("• Check for CPU bottleneck in data loading")
        elif gpu_util < 70:
            issues.append("⚠️  GPU utilization is MODERATE (40-70%)")
            recommendations.append("• Consider increasing batch_size")
            recommendations.append("• Verify num_workers is optimal")
        else:
            issues.append("✅ GPU utilization is GOOD (>70%)")
        
        # Check GPU memory
        if gpu_mem < 50:
            issues.append(f"💡 GPU memory usage is {gpu_mem:.1f}% - room for larger batch_size")
            recommendations.append("• Try increasing batch_size for better GPU utilization")
        elif gpu_mem > 95:
            issues.append("⚠️  GPU memory is FULL (>95%)")
            recommendations.append("• Consider decreasing batch_size to prevent OOM")
        
        # Check CPU
        if cpu > 90:
            issues.append("⚠️  CPU usage is HIGH (>90%)")
            recommendations.append("• CPU may be bottleneck for data loading")
            recommendations.append("• Reduce num_workers if CPU is saturated")
        
        for issue in issues:
            print(f"  {issue}")
        
        if recommendations:
            print("\n  Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
    
    def run(self):
        """Run monitoring loop."""
        print("Starting performance monitor...")
        print("Waiting for training to start...\n")
        
        try:
            while True:
                gpu_metrics = self.get_gpu_metrics()
                cpu_metrics = self.get_cpu_ram_metrics()
                
                # Store metrics
                self.metrics_history['timestamp'].append(time.time())
                self.metrics_history['gpu_util'].append(gpu_metrics['gpu_util'])
                self.metrics_history['gpu_mem_used'].append(gpu_metrics['gpu_mem_used'])
                self.metrics_history['gpu_mem_total'].append(gpu_metrics['gpu_mem_total'])
                self.metrics_history['cpu_percent'].append(cpu_metrics['cpu_percent'])
                self.metrics_history['ram_percent'].append(cpu_metrics['ram_percent'])
                
                # Print metrics
                self.print_metrics(gpu_metrics, cpu_metrics)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics."""
        if not self.metrics_history['gpu_util']:
            return
        
        print("\n" + "="*80)
        print("MONITORING SUMMARY")
        print("="*80)
        
        avg_gpu_util = sum(self.metrics_history['gpu_util']) / len(self.metrics_history['gpu_util'])
        max_gpu_util = max(self.metrics_history['gpu_util'])
        min_gpu_util = min(self.metrics_history['gpu_util'])
        
        print(f"GPU Utilization - Avg: {avg_gpu_util:.1f}%, Min: {min_gpu_util:.1f}%, Max: {max_gpu_util:.1f}%")
        
        if avg_gpu_util < 70:
            print("\n⚠️  Average GPU utilization is below 70%.")
            print("Consider:")
            print("  • Increasing num_workers (current: check config)")
            print("  • Increasing batch_size (current: check config)")
            print("  • Enabling prefetch_factor and persistent_workers")
        else:
            print("\n✅ Good GPU utilization!")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor training performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to monitor')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if not (NVML_AVAILABLE or GPUTIL_AVAILABLE):
        print("Error: Neither pynvml nor GPUtil is available.")
        print("Install one of them:")
        print("  pip install pynvml")
        print("  pip install gputil")
        sys.exit(1)
    
    monitor = PerformanceMonitor(gpu_id=args.gpu, interval=args.interval)
    monitor.run()


if __name__ == '__main__':
    main()
