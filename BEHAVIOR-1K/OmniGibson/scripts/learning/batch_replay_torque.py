"""Batch driver for physics-on torque replays (Milestone 1 production).

Pure-python orchestrator — no sim imports. For one task at a time:
  1. lists the task's episodes on HF `behavior-1k/2026-challenge-rawdata`
     and downloads any missing raw demos (~5-15 MB each)
  2. groups episodes by scene object-set (chunks must share it) and schedules
     chained chunks of --chunk-size episodes: one Isaac boot + one scene load
     per chunk instead of per episode (~2x throughput)
  3. runs `replay_torque.py --demo-ids ...` chunks in a pool of worker
     subprocesses, one Isaac instance each (13.9 GB VRAM/instance measured ->
     2 workers per RTX 5090)
  4. resumable: done demo_ids are read from output-file attrs (single-episode
     `demo_id` or chunk `manifest`); outputs without attrs are partial chunks
     from a crash and are deleted + redone

Outputs land in <data_folder>/replayed_torque/task-XXXX/ as self-identified
HDF5s (task_id, demo_id(s), replay_cell, proprio_layout attrs).

Usage:
  python batch_replay_torque.py --data_folder ~/b1k-data --task-id 0 --workers 2
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
import urllib.request

HF_API = "https://huggingface.co/api/datasets/behavior-1k/2026-challenge-rawdata/tree/main/task-{task_id:04d}"
HF_FILE = "https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata/resolve/main/{path}"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def list_task_episodes(task_id: int) -> tuple[list[int], float]:
    """Returns (sorted demo_ids, total raw size in GB)."""
    with urllib.request.urlopen(HF_API.format(task_id=task_id), timeout=30) as r:
        entries = json.load(r)
    demo_ids, total = [], 0
    for e in entries:
        name = os.path.basename(e["path"])
        if name.startswith("episode_") and name.endswith(".hdf5"):
            demo_ids.append(int(name[len("episode_") : -len(".hdf5")]))
            total += e.get("size", 0)
    return sorted(demo_ids), total / 1e9


def ensure_raw_demo(data_folder: str, task_id: int, demo_id: int) -> str:
    raw_dir = os.path.join(data_folder, "2026-challenge-rawdata", f"task-{task_id:04d}")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"episode_{demo_id:08d}.hdf5")
    if not os.path.exists(path):
        url = HF_FILE.format(path=f"task-{task_id:04d}/episode_{demo_id:08d}.hdf5")
        tmp = path + ".part"
        # HF rate-limits burst downloads (observed HTTP 429 after ~200 rapid
        # fetches) — pace politely and back off hard on 429/5xx.
        delay = 10.0
        for attempt in range(8):
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, path)
                time.sleep(0.5)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503):
                    raise
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if (retry_after or "").isdigit() else delay
                print(f"HTTP {e.code} on demo {demo_id}, retry {attempt + 1}/8 in {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 320)
        else:
            raise RuntimeError(f"download failed after retries: {url}")
    return path


def scene_key(raw_path: str) -> str:
    """Object-set fingerprint — chained chunks must share it."""
    import h5py

    with h5py.File(raw_path, "r") as f:
        sj = json.loads(f["data"].attrs["scene_file"])
    names = ",".join(sorted(sj["objects_info"]["init_info"].keys()))
    return hashlib.md5(names.encode()).hexdigest()[:10]


def collect_done(out_dir: str) -> set[int]:
    """Done demo_ids from output attrs; partial (attr-less) files are removed."""
    import h5py

    done = set()
    for name in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if not name.endswith(".hdf5"):
            continue
        path = os.path.join(out_dir, name)
        try:
            with h5py.File(path, "r") as f:
                attrs = f["data"].attrs
                if "manifest" in attrs:
                    done.update(e["demo_id"] for e in json.loads(attrs["manifest"]))
                elif "demo_id" in attrs:
                    done.add(int(attrs["demo_id"]))
                else:
                    raise KeyError("no identity attrs")
        except Exception:
            print(f"removing partial/corrupt output {name}", flush=True)
            os.remove(path)
    return done


def chunk_complete(out: str, chunk: list[int]) -> bool:
    """Content-based success check. Isaac's shutdown handler can exit 0 even
    after a crash, so the subprocess returncode proves nothing — only a
    manifest covering every requested demo_id does."""
    import h5py

    try:
        with h5py.File(out, "r") as f:
            manifest = json.loads(f["data"].attrs["manifest"])
        return {e["demo_id"] for e in manifest} == set(chunk)
    except Exception:
        return False


def run_chunk(data_folder: str, task_id: int, chunk: list[int], log_dir: str,
              gpu_slots: "queue.Queue[str]") -> tuple[list[int], bool, float]:
    out = os.path.join(data_folder, "replayed_torque", f"task-{task_id:04d}",
                       f"chunk_{chunk[0]:08d}_{len(chunk)}.hdf5")
    t0 = time.time()
    gpu = gpu_slots.get()
    try:
        threads = os.environ.get("REPLAY_PHYSICS_THREADS", "6")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES",
                   # torch/OpenMP default to all-cores intra-op pools PER worker
                   # (observed: 12+ threads at 40-50% each, load 67 on 24 cores)
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads)
        # timeout scales with raw size: measured ~0.15 min/MB/worker, x3 margin
        # (fixed formula killed healthy long-episode chunks — task 1, 25 MB eps)
        raw_dir = os.path.join(data_folder, "2026-challenge-rawdata", f"task-{task_id:04d}")
        chunk_mb = sum(
            os.path.getsize(os.path.join(raw_dir, f"episode_{d:08d}.hdf5")) for d in chunk
        ) / 1e6
        timeout_s = 900 + int(chunk_mb * 30)
        log_path = os.path.join(log_dir, f"chunk_{chunk[0]:08d}_{len(chunk)}.log")
        try:
            with open(log_path, "w") as log_f:
                subprocess.run(
                    [sys.executable, os.path.join(SCRIPT_DIR, "replay_torque.py"),
                     "--data_folder", data_folder,
                     "--demo-ids", ",".join(str(d) for d in chunk),
                     "--output", out,
                     # lean validated bit-exact vs full replay (X1, 2026-07-18):
                     # 13.9 -> 2.7 GB VRAM, ~20% faster, corr 1.00000
                     "--lean",
                     "--physics-threads", os.environ.get("REPLAY_PHYSICS_THREADS", "0")],
                    stdout=log_f, stderr=subprocess.STDOUT, env=env, timeout=timeout_s,
                )
        except subprocess.TimeoutExpired:
            print(f"chunk {chunk[0]}..{chunk[-1]}: TIMEOUT after {timeout_s}s", flush=True)
    finally:
        gpu_slots.put(gpu)
    ok = chunk_complete(out, chunk)
    if not ok and os.path.exists(out):
        os.remove(out)  # partial chunk — redo whole on next resume
    return chunk, ok, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2, help="Workers PER GPU")
    parser.add_argument("--gpus", type=str, default=os.environ.get("REPLAY_GPU", "0"),
                        help="Comma-separated GPU ids to spread workers over")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N episodes (0 = all)")
    parser.add_argument("--delete-raw-after", action="store_true",
                        help="Delete the task's raw demo downloads after a fully-successful run "
                        "(sidecars are kept) — bounds disk usage to ~one task of raw data")
    parser.add_argument("--max-raw-gb", type=float, default=0,
                        help="Skip the task (exit 3) if its raw demos exceed this size — leave "
                        "oversized tasks for machines with more disk (0 = no limit)")
    args = parser.parse_args()

    data_folder = os.path.expanduser(args.data_folder)
    out_dir = os.path.join(data_folder, "replayed_torque", f"task-{args.task_id:04d}")
    log_dir = os.path.join(data_folder, "replay_logs", f"task-{args.task_id:04d}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    demo_ids, raw_gb = list_task_episodes(args.task_id)
    if args.max_raw_gb and raw_gb > args.max_raw_gb:
        print(f"task {args.task_id}: raw size {raw_gb:.1f} GB exceeds --max-raw-gb "
              f"{args.max_raw_gb} — skipping (leave for a bigger machine)", flush=True)
        sys.exit(3)
    if args.limit:
        demo_ids = demo_ids[: args.limit]
    done = collect_done(out_dir)
    todo = [d for d in demo_ids if d not in done]
    print(f"task {args.task_id}: {len(demo_ids)} episodes, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    # download everything first (tiny files), then group by scene object-set
    by_key: dict[str, list[int]] = {}
    for d in todo:
        key = scene_key(ensure_raw_demo(data_folder, args.task_id, d))
        by_key.setdefault(key, []).append(d)
    chunks = []
    for key, ids in sorted(by_key.items()):
        for i in range(0, len(ids), args.chunk_size):
            chunks.append(ids[i : i + args.chunk_size])
    gpus = args.gpus.split(",")
    n_workers = args.workers * len(gpus)
    if "REPLAY_PHYSICS_THREADS" not in os.environ:
        # avoid PhysX thread-pool oversubscription: each Isaac instance defaults
        # to an all-cores dispatcher (observed: load 69 on 24 cores at 4 workers)
        os.environ["REPLAY_PHYSICS_THREADS"] = str(max(2, (os.cpu_count() or 8) // max(n_workers, 1) - 1))
        print(f"physics threads per worker: {os.environ['REPLAY_PHYSICS_THREADS']}", flush=True)
    gpu_slots: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        for _ in range(args.workers):
            gpu_slots.put(g)
    print(f"{len(by_key)} scene group(s) -> {len(chunks)} chunk(s) of <= {args.chunk_size}, "
          f"{n_workers} workers on GPU(s) {args.gpus}", flush=True)

    done_n = failed_n = 0
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_chunk, data_folder, args.task_id, c, log_dir, gpu_slots): c for c in chunks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                chunk, ok, dt = fut.result()
            except Exception as e:
                chunk, ok, dt = futures[fut], False, 0.0
                print(f"chunk {chunk[0]}..{chunk[-1]}: worker exception {type(e).__name__}: {e}", flush=True)
            done_n += len(chunk) * ok
            failed_n += len(chunk) * (not ok)
            eph = (done_n + failed_n) / max(time.time() - t_start, 1) * 3600
            print(f"chunk {chunk[0]}..{chunk[-1]} ({len(chunk)} eps): {'OK' if ok else 'FAILED'} "
                  f"({dt / 60:.1f} min) — {done_n} ok / {failed_n} failed / {len(todo)} total, "
                  f"{eph:.0f} eps/h", flush=True)

    print(f"DONE task {args.task_id}: {done_n} ok, {failed_n} failed, "
          f"{(time.time() - t_start) / 3600:.2f} h", flush=True)
    if args.delete_raw_after and failed_n == 0:
        raw_dir = os.path.join(data_folder, "2026-challenge-rawdata", f"task-{args.task_id:04d}")
        if os.path.isdir(raw_dir):
            import shutil

            shutil.rmtree(raw_dir)
            print(f"deleted raw demos: {raw_dir} ({raw_gb:.1f} GB freed)", flush=True)
    sys.exit(1 if failed_n else 0)


if __name__ == "__main__":
    main()
