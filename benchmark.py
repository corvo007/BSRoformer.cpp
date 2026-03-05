#!/usr/bin/env python3
import subprocess
import time
import psutil
import threading
import os
import csv
from datetime import datetime

# Configuration
MODELS = [
    {"name": "voc_fv6", "path": r"D:\Download\voc_fv6-Q8_0.gguf"},
    {"name": "becruily_deux", "path": r"D:\Download\becruily_deux-Q8_0.gguf"}
]

AUDIOS = [
    {"name": "5min", "path": "test_song.wav"},
    {"name": "24min", "path": "test_23min.wav"}
]

BACKENDS = [
    {"name": "CUDA-Optimized", "exe": r"build-cuda\Release\bs_roformer-cli.exe"},
    {"name": "Vulkan-Optimized", "exe": r"build-vulkan\Release\bs_roformer-cli.exe"}
]

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False
        self.process = None

    def monitor(self, pid):
        self.monitoring = True
        try:
            proc = psutil.Process(pid)
            while self.monitoring:
                try:
                    mem_info = proc.memory_info()
                    current_mem = mem_info.rss / (1024 * 1024)  # MB
                    self.peak_memory = max(self.peak_memory, current_mem)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass

    def stop(self):
        self.monitoring = False

def run_benchmark(backend, model, audio):
    output_file = f"benchmark_{backend['name']}_{model['name']}_{audio['name']}.wav"

    cmd = [backend['exe'], model['path'], audio['path'], output_file]

    monitor = MemoryMonitor()

    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Start memory monitoring in separate thread
    monitor_thread = threading.Thread(target=monitor.monitor, args=(process.pid,))
    monitor_thread.start()

    # Capture output
    output_lines = []
    for line in process.stdout:
        output_lines.append(line.strip())

    process.wait()
    end_time = time.time()

    monitor.stop()
    monitor_thread.join()

    # Extract processing time from output
    process_time = None
    for line in output_lines:
        if "Processed in" in line:
            try:
                process_time = float(line.split("Processed in")[1].split("seconds")[0].strip())
            except:
                pass

    if process_time is None:
        process_time = end_time - start_time

    # Clean up output file
    generated_paths = set()
    for line in output_lines:
        if "Saved output stem" in line or "Saving output stem" in line:
            _, _, tail = line.partition(":")
            path = tail.strip()
            if path:
                generated_paths.add(path)

    if not generated_paths:
        generated_paths.add(output_file)

        # Best-effort cleanup for multi-stem outputs (CLI uses *_stem_N.wav naming).
        root, ext = os.path.splitext(output_file)
        for i in range(16):
            generated_paths.add(f"{root}_stem_{i}{ext}")

    for path in generated_paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    return {
        "backend": backend['name'],
        "model": model['name'],
        "audio": audio['name'],
        "process_time": round(process_time, 2),
        "peak_memory_mb": round(monitor.peak_memory, 2)
    }

def main():
    print("=== MelBandRoformer Benchmark ===\n")

    results = []

    for backend in BACKENDS:
        for model in MODELS:
            for audio in AUDIOS:
                test_name = f"{backend['name']} | {model['name']} | {audio['name']}"
                print(f"Running: {test_name}")

                try:
                    result = run_benchmark(backend, model, audio)
                    results.append(result)
                    print(f"  Time: {result['process_time']}s, Peak Memory: {result['peak_memory_mb']} MB\n")
                except Exception as e:
                    print(f"  Error: {e}\n")

    # Save results
    csv_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['backend', 'model', 'audio', 'process_time', 'peak_memory_mb'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Results saved to {csv_file} ===")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Backend':<20} {'Model':<15} {'Audio':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['backend']:<20} {r['model']:<15} {r['audio']:<10} {r['process_time']:<10} {r['peak_memory_mb']:<12}")

if __name__ == "__main__":
    main()
