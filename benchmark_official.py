#!/usr/bin/env python3
import subprocess
import time
import psutil
import threading
import os
import csv
from datetime import datetime

# Configuration - Official versions
MODELS = [
    {"name": "voc_fv6", "path": r"D:\Download\voc_fv6-Q8_0.gguf"},
    {"name": "becruily_deux", "path": r"D:\Download\becruily_deux-Q8_0.gguf"}
]

AUDIOS = [
    {"name": "5min", "path": r"D:\onedrive\codelab\Gemini-Subtitle-Pro\MelBandRoformer.cpp\test_song.wav"},
    {"name": "24min", "path": r"D:\onedrive\codelab\Gemini-Subtitle-Pro\MelBandRoformer.cpp\test_23min.wav"}
]

BACKENDS = [
    {"name": "CUDA-Official", "exe": r"D:\onedrive\codelab\Gemini-Subtitle-Pro\MelBandRoformer.cpp-official\build-cuda\Release\bs_roformer-cli.exe"}
]

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False

    def monitor(self, pid):
        self.monitoring = True
        try:
            proc = psutil.Process(pid)
            while self.monitoring:
                try:
                    mem_info = proc.memory_info()
                    current_mem = mem_info.rss / (1024 * 1024)
                    self.peak_memory = max(self.peak_memory, current_mem)
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass

    def stop(self):
        self.monitoring = False

def run_benchmark(backend, model, audio):
    output_file = f"benchmark_official_{backend['name']}_{model['name']}_{audio['name']}.wav"

    cmd = [backend['exe'], model['path'], audio['path'], output_file]

    monitor = MemoryMonitor()

    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    monitor_thread = threading.Thread(target=monitor.monitor, args=(process.pid,))
    monitor_thread.start()

    output_lines = []
    for line in process.stdout:
        output_lines.append(line.strip())

    process.wait()
    end_time = time.time()

    monitor.stop()
    monitor_thread.join()

    process_time = None
    for line in output_lines:
        if "Processed in" in line:
            try:
                process_time = float(line.split("Processed in")[1].split("seconds")[0].strip())
            except:
                pass

    if process_time is None:
        process_time = end_time - start_time

    if os.path.exists(output_file):
        os.remove(output_file)

    return {
        "backend": backend['name'],
        "model": model['name'],
        "audio": audio['name'],
        "process_time": round(process_time, 2),
        "peak_memory_mb": round(monitor.peak_memory, 2)
    }

def main():
    print("=== Official Version Benchmark ===\n")

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

    csv_file = f"benchmark_official_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['backend', 'model', 'audio', 'process_time', 'peak_memory_mb'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Results saved to {csv_file} ===")

    print("\n=== Summary ===")
    print(f"{'Backend':<20} {'Model':<15} {'Audio':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['backend']:<20} {r['model']:<15} {r['audio']:<10} {r['process_time']:<10} {r['peak_memory_mb']:<12}")

if __name__ == "__main__":
    main()
