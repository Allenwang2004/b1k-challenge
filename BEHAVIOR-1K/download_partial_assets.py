"""
Selective downloader for behavior-1k-assets: only pulls the object models needed
for the house_double_floor_lower scene (turning_on_radio task), instead of the
full 29.3GB dataset zip. Uses HTTP range requests (remotezip) against the same
HF-hosted zip that omnigibson.utils.asset_utils.download_behavior_1k_assets()
would otherwise download in full.

The HF CDN signs the redirect URL with a short-lived (~15-20 min) expiry, which
is not enough to stream thousands of small files sequentially, so this refreshes
the URL (and reopens RemoteZip) whenever a fetch fails, resuming from the same
file (already-extracted files are skipped via an on-disk size check).

Run inside the b1k-sim container (needs `omnigibson` importable + `remotezip`
pip-installed), with the repo's datasets/ dir mounted at /data.
"""
import json
import os
import time

import omnigibson.utils.asset_utils as au
import requests
from remotezip import RemoteZip

REPO_ID = "behavior-1k/zipped-datasets"
ZIP_FILENAME = f"behavior-1k-assets-{au.BsEHAVIOR_1K_ASSET_VERSION}.zip"

SCENE_MODEL = "house_double_floor_lower"
TASK_NAME = "turning_on_radio"


def get_remote_zip_url() -> str:
    resolve_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ZIP_FILENAME}"
    r = requests.head(resolve_url, allow_redirects=False, timeout=30)
    return r.headers["location"]


def collect_needed_pairs() -> set[tuple[str, str]]:
    data_path = au.gm.DATA_PATH
    base_scene_json = os.path.join(
        data_path, "2026-challenge-task-instances", "scenes", SCENE_MODEL, "json", f"{SCENE_MODEL}_stable.json"
    )
    template_json = os.path.join(
        data_path,
        "2026-challenge-task-instances",
        "scenes",
        SCENE_MODEL,
        "json",
        f"{SCENE_MODEL}_task_{TASK_NAME}_0_0_template.json",
    )

    pairs = set()
    for path in (base_scene_json, template_json):
        with open(path) as f:
            d = json.load(f)
        for _, info in d["objects_info"]["init_info"].items():
            args = info["args"]
            cat, model = args.get("category"), args.get("model")
            if cat and model:
                pairs.add((cat, model))
    return pairs


def already_extracted(target_root: str, info) -> bool:
    fpath = os.path.join(target_root, info.filename)
    return os.path.exists(fpath) and not info.is_dir() and os.path.getsize(fpath) == info.file_size


def main():
    key_path = au.get_key_path()
    if os.path.exists(key_path):
        print(f"Decryption key already present at {key_path}")
    else:
        print("Downloading BEHAVIOR-1K decryption key...")
        au.download_key()
        print(f"Key written to {key_path}")

    pairs = collect_needed_pairs()
    print(f"Need {len(pairs)} unique (category, model) object pairs for {TASK_NAME} in {SCENE_MODEL}")

    target_root = au.get_dataset_path("behavior-1k-assets")
    os.makedirs(target_root, exist_ok=True)

    prefixes = tuple(f"objects/{c}/{m}/" for c, m in pairs)
    prefixes += (f"scenes/{SCENE_MODEL}/", "metadata/", "systems/")

    z = RemoteZip(get_remote_zip_url())
    infos = z.infolist()
    to_fetch = [i for i in infos if i.filename.startswith(prefixes) or i.filename == "VERSION"]
    total_bytes = sum(i.file_size for i in to_fetch)
    print(f"Fetching {len(to_fetch)} files, {total_bytes / 1e9:.3f} GB uncompressed...")

    done_bytes = sum(i.file_size for i in to_fetch if already_extracted(target_root, i))
    skipped = sum(1 for i in to_fetch if already_extracted(target_root, i))
    print(f"Resuming: {skipped}/{len(to_fetch)} files already on disk ({done_bytes / 1e9:.3f} GB)")

    reconnects = 0
    for idx, info in enumerate(to_fetch):
        if already_extracted(target_root, info):
            continue
        while True:
            try:
                z.extract(info, path=target_root)
                break
            except Exception as e:
                reconnects += 1
                print(f"  [retry {reconnects}] fetch failed at file {idx} ({info.filename}): {e}")
                time.sleep(2)
                z = RemoteZip(get_remote_zip_url())
        done_bytes += info.file_size
        if idx % 500 == 0 or idx == len(to_fetch) - 1:
            print(f"  [{idx + 1}/{len(to_fetch)}] {done_bytes / 1e9:.3f} / {total_bytes / 1e9:.3f} GB")

    print(f"Done extracting partial behavior-1k-assets ({reconnects} URL refreshes needed).")

    print("Downloading omnigibson-robot-assets (full, ~611MB)...")
    au.download_omnigibson_robot_assets()
    print("All done.")


if __name__ == "__main__":
    main()
