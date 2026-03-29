#!/usr/bin/env python3
"""
BSRoformer Vulkan Optimization Benchmark
Multi-objective evaluation: Speed (RTF) + Memory + Quality (SNR)

Usage:
    python benchmark.py baseline       # Establish baseline (saves reference WAVs)
    python benchmark.py test [label]   # Test current build against baseline
    python benchmark.py full [label]   # Full test including vlog
"""
import subprocess
import time
import psutil
import threading
import os
import sys
import struct
import math
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
CLI = ROOT / "build-vulkan" / "Release" / "bs-roformer-cli.exe"
OUT_DIR = ROOT / "benchmark_out"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "voc": str(ROOT / "model" / "voc_fv6-Q8_0.gguf"),
    "becruily": str(ROOT / "model" / "becruily_deux-Q8_0.gguf"),
}
AUDIOS = {
    "song": str(ROOT / "test_song.wav"),
    "vlog": str(ROOT / "test_23min.wav"),
}

# Quality threshold: if SNR drops below this, the change is rejected
MIN_SNR_DB = 60.0  # Very high SNR required (essentially bit-exact)


class MemoryMonitor:
    def __init__(self):
        self.peak_memory_mb = 0.0
        self._stop = False

    def monitor(self, pid):
        try:
            proc = psutil.Process(pid)
            while not self._stop:
                try:
                    mem = proc.memory_info().rss / (1024 * 1024)
                    self.peak_memory_mb = max(self.peak_memory_mb, mem)
                    time.sleep(0.05)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass

    def stop(self):
        self._stop = True


def wav_duration(path: str) -> float:
    with open(path, "rb") as f:
        f.read(12)  # RIFF + size + WAVE
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt = f.read(chunk_size)
                channels = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                bps = struct.unpack("<H", fmt[14:16])[0]
            elif chunk_id == b"data":
                return chunk_size / (channels * (bps // 8)) / sr
            else:
                f.seek(chunk_size, 1)
    return 0.0


def read_wav_samples(path: str, max_frames: int = 44100 * 30):
    """Read first max_frames of WAV as float32 list."""
    with open(path, "rb") as f:
        f.read(12)
        fmt = None
        while True:
            cid = f.read(4)
            if len(cid) < 4:
                return []
            csz = struct.unpack("<I", f.read(4))[0]
            if cid == b"fmt ":
                fmt = f.read(csz)
            elif cid == b"data":
                audio_fmt = struct.unpack("<H", fmt[0:2])[0]
                ch = struct.unpack("<H", fmt[2:4])[0]
                bps = struct.unpack("<H", fmt[14:16])[0]
                n_total = csz // (ch * (bps // 8))
                n = min(n_total, max_frames) * ch
                if audio_fmt == 3:  # float32
                    raw = f.read(n * 4)
                    return list(struct.unpack(f"<{n}f", raw))
                elif audio_fmt == 1 and bps == 16:
                    raw = f.read(n * 2)
                    return [s / 32768.0 for s in struct.unpack(f"<{n}h", raw)]
                elif audio_fmt == 1 and bps == 32:
                    raw = f.read(n * 4)
                    return [s / 2147483648.0 for s in struct.unpack(f"<{n}i", raw)]
                return []
            else:
                f.seek(csz, 1)
    return []


def compare_quality(ref_path: str, test_path: str) -> dict:
    """Compare two WAV files. Returns SNR, L2, max_diff."""
    ref = read_wav_samples(ref_path)
    test = read_wav_samples(test_path)
    n = min(len(ref), len(test))
    if n == 0:
        return {"snr_db": 0.0, "l2": 999.0, "max_diff": 999.0, "n": 0}

    ref = ref[:n]
    test = test[:n]

    sum_sig = sum(a * a for a in ref)
    sum_noise = sum((a - b) ** 2 for a, b in zip(ref, test))
    max_diff = max(abs(a - b) for a, b in zip(ref, test))

    rms_sig = math.sqrt(sum_sig / n)
    rms_noise = math.sqrt(sum_noise / n)
    snr = 20 * math.log10(rms_sig / max(rms_noise, 1e-15))

    return {
        "snr_db": round(snr, 1),
        "l2": round(rms_noise, 8),
        "max_diff": round(max_diff, 6),
        "n": n,
    }


def run_one(tag: str, model: str, audio: str, extra_args=None) -> dict:
    """Run one inference and collect metrics."""
    out_wav = str(OUT_DIR / f"{tag}.wav")
    stdout_f = str(OUT_DIR / f"{tag}_stdout.txt")
    stderr_f = str(OUT_DIR / f"{tag}_stderr.txt")

    cmd = [str(CLI), model, audio, out_wav]
    if extra_args:
        cmd.extend(extra_args)

    dur = wav_duration(audio)
    print(f"\n{'='*60}")
    print(f"  {tag}  (audio={dur:.0f}s)")
    print(f"  {' '.join(cmd)}")

    mon = MemoryMonitor()
    with open(stdout_f, "w") as fo, open(stderr_f, "w") as fe:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe)
        mt = threading.Thread(target=mon.monitor, args=(proc.pid,), daemon=True)
        mt.start()
        proc.wait()
        t1 = time.perf_counter()
        mon.stop()
        mt.join(timeout=1)

    elapsed = t1 - t0
    rtf = elapsed / dur if dur > 0 else 999.0
    ok = proc.returncode == 0

    print(f"  {'OK' if ok else 'FAIL'} | {elapsed:.1f}s | RTF={rtf:.4f} | Mem={mon.peak_memory_mb:.0f}MB")

    return {
        "tag": tag,
        "model": Path(model).stem,
        "audio": Path(audio).stem,
        "duration_s": round(dur, 1),
        "time_s": round(elapsed, 1),
        "rtf": round(rtf, 4),
        "peak_mem_mb": round(mon.peak_memory_mb, 0),
        "exit_code": proc.returncode,
        "out_wav": out_wav,
    }


def run_suite(label: str, audio_keys=None, extra_args=None):
    """Run all model x audio combinations."""
    if audio_keys is None:
        audio_keys = ["song"]  # Default: fast iteration
    results = []
    for mk, mp in MODELS.items():
        for ak in audio_keys:
            ap = AUDIOS[ak]
            tag = f"{label}_{mk}_{ak}"
            r = run_one(tag, mp, ap, extra_args)
            results.append(r)
    return results


def evaluate(results: list, baseline_label="baseline"):
    """Compare results against baseline. Print unified score."""
    print(f"\n{'='*80}")
    print(f"  EVALUATION")
    print(f"{'='*80}")
    print(f"{'Tag':<35} {'Time':>7} {'RTF':>8} {'Mem':>7} {'SNR':>8} {'Verdict':>8}")
    print("-" * 80)

    all_ok = True
    rtf_improvements = []
    mem_improvements = []

    for r in results:
        # Match baseline by model+audio (tag format: {label}_{model}_{audio})
        model_key = r["model"].replace("-Q8_0", "").replace("_deux", "")  # normalize
        audio_key = r["audio"]
        # Find baseline with matching model and audio
        bl_tag = None
        bl_json = OUT_DIR / "baseline_results.json"
        bl_results = []
        if bl_json.exists():
            with open(bl_json) as f:
                bl_results = json.load(f)
        for b in bl_results:
            if b["model"] == r["model"] and b["audio"] == r["audio"]:
                bl_tag = b["tag"]
                break

        snr_str = "N/A"
        verdict = "OK"

        # Quality check
        bl_wav = str(OUT_DIR / f"{bl_tag}.wav") if bl_tag else None
        if bl_wav and os.path.exists(bl_wav) and os.path.exists(r["out_wav"]):
            q = compare_quality(bl_wav, r["out_wav"])
            snr_str = f"{q['snr_db']:.0f}dB"
            if q["snr_db"] < MIN_SNR_DB:
                verdict = "REGRESS"
                all_ok = False

        # Speed/memory comparison
        bl_match = [b for b in bl_results if bl_tag and b["tag"] == bl_tag]
        if bl_match:
            bl = bl_match[0]
            rtf_imp = (bl["rtf"] - r["rtf"]) / bl["rtf"] * 100
            mem_imp = (bl["peak_mem_mb"] - r["peak_mem_mb"]) / bl["peak_mem_mb"] * 100 if bl["peak_mem_mb"] > 0 else 0
            rtf_improvements.append(rtf_imp)
            mem_improvements.append(mem_imp)

        if r["exit_code"] != 0:
            verdict = "CRASH"
            all_ok = False

        print(f"{r['tag']:<35} {r['time_s']:>6.1f}s {r['rtf']:>8.4f} {r['peak_mem_mb']:>6.0f}M {snr_str:>8} {verdict:>8}")

    # Composite score
    if rtf_improvements:
        avg_rtf = sum(rtf_improvements) / len(rtf_improvements)
        avg_mem = sum(mem_improvements) / len(mem_improvements) if mem_improvements else 0
        score = avg_rtf + 0.5 * avg_mem
        print(f"\nComposite: RTF_imp={avg_rtf:+.1f}%  Mem_imp={avg_mem:+.1f}%  Score={score:+.1f}")
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python benchmark.py baseline           # Establish baseline")
        print("  python benchmark.py test [label]        # Quick test (song only)")
        print("  python benchmark.py full [label]        # Full test (song + vlog)")
        return

    mode = sys.argv[1]

    if mode == "baseline":
        results = run_suite("baseline", ["song"])
        with open(OUT_DIR / "baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nBaseline saved.")
        for r in results:
            print(f"  {r['tag']}: RTF={r['rtf']} Time={r['time_s']}s Mem={r['peak_mem_mb']}MB")

    elif mode == "test":
        label = sys.argv[2] if len(sys.argv) > 2 else f"exp_{datetime.now().strftime('%H%M')}"
        results = run_suite(label, ["song"])
        evaluate(results)

    elif mode == "full":
        label = sys.argv[2] if len(sys.argv) > 2 else f"full_{datetime.now().strftime('%H%M')}"
        results = run_suite(label, ["song", "vlog"])
        evaluate(results)

    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
