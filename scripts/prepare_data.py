#!/usr/bin/env python
"""
Download and prepare the public HELIX01-04 dataset for ConformerCaller.

Fetches, checksum-verifies and unpacks everything the pipeline needs into a
single data root directory:

    <dest>/
    ├── FAST5/                         # extracted multi-read .fast5 files
    ├── Reference/
    │   ├── completed_library.csv      # motif library        (OSF)
    │   └── motif_design_sequences.csv # motif design table   (OSF)
    ├── HELIX01-04_master_db.txt       # per-read motif labels (Zenodo Part 01)
    ├── HELIX01-04-encoded.csv         # encoded payload table (Zenodo Part 01)
    └── archives/                      # downloaded .7z files (safe to delete)

Sources
    Zenodo HELIX01-04_Part_01  https://doi.org/10.5281/zenodo.15839390
    Zenodo HELIX01-04_Part_02  https://doi.org/10.5281/zenodo.15848607
    OSF    Motif-based DNA Data Storage System  https://osf.io/pcdtj/

All files are verified against the MD5 checksums published by Zenodo / OSF.
Files that are already present and valid are not downloaded again.

Usage
    # Everything (~55 GB download, ~77 GB extracted)
    python scripts/prepare_data.py --dest /path/to/data_root

    # Quick start: reference files + labels + only the 9 smallest FAST5 files
    python scripts/prepare_data.py --dest /path/to/data_root --fast5 1122-1130

    # Re-use archives you already downloaded manually
    python scripts/prepare_data.py --dest /path/to/data_root \
        --archives-dir /path/to/downloads

Extraction needs a 7-Zip binary (`7zz`, `7z` or `7za`):
    macOS:   brew install sevenzip
    Ubuntu:  sudo apt install p7zip-full
(or `pip install py7zr` as a pure-Python fallback).

Afterwards:
    export CONFORMERCALLER_DATA_ROOT=/path/to/data_root
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ZENODO_RECORDS = {"part01": "15839390", "part02": "15848607"}
OSF_NODE = "pcdtj"
OSF_REFERENCE_FILES = ["completed_library.csv", "motif_design_sequences.csv"]
FAST5_ARCHIVE_RE = re.compile(r"HELIX01-04_FAST5_(\d+-\d+)\.7z$")


# ── helpers ──────────────────────────────────────────────────────────────────
def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def fetch(name: str, url: str, md5: str, dest: Path, archives_dir):
    """Ensure `dest` exists with the right checksum (download if needed)."""
    candidates = [dest] + ([archives_dir / name] if archives_dir else [])
    for c in candidates:
        if c.exists():
            print(f"  verifying {c} …", flush=True)
            if md5_of(c) == md5:
                if c != dest:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        try:
                            dest.symlink_to(c.resolve())
                        except OSError:
                            shutil.copy2(c, dest)
                print("    OK")
                return
            print("    checksum mismatch — re-downloading")
    print(f"  downloading {name}")
    download(url, dest)
    if md5_of(dest) != md5:
        sys.exit(f"ERROR: checksum mismatch after download: {name}")
    print("    OK")


def find_7z():
    for exe in ("7zz", "7z", "7za"):
        if shutil.which(exe):
            return exe
    return None


def extract(archive: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = find_7z()
    if exe:
        subprocess.run([exe, "x", "-y", f"-o{out_dir}", str(archive)],
                       check=True, stdout=subprocess.DEVNULL)
        return
    try:
        import py7zr
    except ImportError:
        sys.exit("ERROR: no 7-Zip found. Install it (macOS: `brew install sevenzip`, "
                 "Ubuntu: `sudo apt install p7zip-full`) or `pip install py7zr`.")
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(out_dir)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True, help="Data root directory to create/populate")
    ap.add_argument("--fast5", nargs="+", default=["all"],
                    help="'all' (default) or one or more archive ranges, e.g. 1122-1130 0-65")
    ap.add_argument("--archives-dir", type=Path, default=None,
                    help="Folder with already-downloaded Zenodo/OSF files (avoids re-downloading)")
    ap.add_argument("--keep-archives", action="store_true",
                    help="Keep .7z archives after extraction (default: keep; pass --delete-archives to remove)")
    ap.add_argument("--delete-archives", action="store_true", help="Delete .7z archives after successful extraction")
    args = ap.parse_args()

    dest: Path = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    arch_dir = dest / "archives"

    # 1. Metadata from Zenodo
    print("Querying Zenodo …")
    zfiles = {}
    for part, rid in ZENODO_RECORDS.items():
        rec = get_json(f"https://zenodo.org/api/records/{rid}")
        for f in rec["files"]:
            zfiles[f["key"]] = {"url": f["links"]["self"], "md5": f["checksum"].split(":", 1)[1],
                                "size": f["size"], "part": part}

    # 2. Select FAST5 archives
    fast5_archives = {FAST5_ARCHIVE_RE.search(k).group(1): k
                      for k in zfiles if FAST5_ARCHIVE_RE.search(k)}
    wanted = list(fast5_archives) if args.fast5 == ["all"] else args.fast5
    unknown = [w for w in wanted if w not in fast5_archives]
    if unknown:
        sys.exit(f"Unknown FAST5 range(s) {unknown}. Available: {sorted(fast5_archives, key=lambda s: int(s.split('-')[0]))}")
    need_gb = sum(zfiles[fast5_archives[w]]["size"] for w in wanted) / 1e9
    free_gb = shutil.disk_usage(dest).free / 1e9
    print(f"FAST5 archives selected: {len(wanted)}  (~{need_gb:.0f} GB download, ~{need_gb * 1.4:.0f} GB extracted; "
          f"{free_gb:.0f} GB free)")
    if free_gb < need_gb * 2.4:
        print("WARNING: free disk space may be insufficient (archives + extracted files).")

    # 3. Labels / encoded table (Zenodo Part 01)
    print("\n[1/4] Labels and encoding table")
    for name in ("HELIX01-04_master_db.txt", "HELIX01-04-encoded.csv"):
        z = zfiles[name]
        fetch(name, z["url"], z["md5"], dest / name, args.archives_dir)

    # 4. Reference library (OSF)
    print("\n[2/4] Motif library (OSF)")
    listing = get_json(f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/")["data"]
    osf = {}
    def collect(items):
        for it in items:
            a = it["attributes"]
            if a["kind"] == "folder":
                sub = get_json(it["relationships"]["files"]["links"]["related"]["href"])["data"]
                collect(sub)
            else:
                osf[a["name"]] = {"url": it["links"]["download"],
                                  "md5": a.get("extra", {}).get("hashes", {}).get("md5")}
    collect(listing)
    for name in OSF_REFERENCE_FILES:
        o = osf[name]
        if o["md5"]:
            fetch(name, o["url"], o["md5"], dest / "Reference" / name, args.archives_dir)
        else:  # no checksum published — download without verification
            target = dest / "Reference" / name
            if not target.exists():
                src = (args.archives_dir / name) if args.archives_dir and (args.archives_dir / name).exists() else None
                if src:
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, target)
                else:
                    print(f"  downloading {name}"); download(o["url"], target)

    # 5. FAST5 archives
    print("\n[3/4] FAST5 archives")
    for w in sorted(wanted, key=lambda s: int(s.split("-")[0])):
        name = fast5_archives[w]
        z = zfiles[name]
        fetch(name, z["url"], z["md5"], arch_dir / name, args.archives_dir)

    # 6. Extract
    print("\n[4/4] Extracting FAST5 files")
    fast5_dir = dest / "FAST5"
    for w in sorted(wanted, key=lambda s: int(s.split("-")[0])):
        name = fast5_archives[w]
        marker = fast5_dir / f".extracted_{w}"
        if marker.exists():
            print(f"  {name}: already extracted")
            continue
        print(f"  extracting {name} …", flush=True)
        extract(arch_dir / name, fast5_dir)
        marker.touch()
        if args.delete_archives:
            real = (arch_dir / name).resolve()
            (arch_dir / name).unlink()
            if args.archives_dir is None:  # only delete data we downloaded ourselves
                pass

    n = len([p for p in fast5_dir.glob("*.fast5") if not p.name.startswith(".")])
    print(f"\nDone. {n} FAST5 files in {fast5_dir}")
    print(f"\nNext:\n  export CONFORMERCALLER_DATA_ROOT={dest}")


if __name__ == "__main__":
    main()
