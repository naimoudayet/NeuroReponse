#!/usr/bin/env python3
"""Resumable downloader for the full TDBRAIN dataset zip.

The brainclinics/Cloudflare server drops the connection partway through the
~14.5 GB download (it tends to die around the same offset every time). The
server *does* honor HTTP range requests (verified: a ranged GET returns
``206 Partial Content`` with a correct ``Content-Range``), so this script
downloads with a ``Range`` header and, on every drop or stall, resumes from
the current size of the partial file.

Just run it. If it stops, run it again - it picks up where it left off.
It also auto-retries in a loop, so normally you only launch it once.

    python download_tdbrain.py
    python download_tdbrain.py --out D:/somewhere/TDBRAIN.zip   # custom path

Notes
-----
* The file is ``..._Encr.zip`` = encrypted. Downloading it is what this does;
  you still need the password from brainclinics to unzip it afterwards.
* Verify the final size is exactly 15544720874 bytes (printed at the end).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

# Windows consoles default to cp1252, which crashes on non-ASCII output. Force
# UTF-8 and never let an un-encodable char kill the download mid-flight.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

URL = "https://brainclinics.com/images/downloads/TDBRAIN_Dataset_V3_1_Encr.zip"
DEFAULT_OUT = "data/tdbrain/TDBRAIN_Dataset_V3_1_Encr.zip"
CHUNK = 1024 * 1024  # 1 MiB per write
# Browser-like UA - Cloudflare sometimes throttles the default python-requests UA.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120  # a stalled read longer than this triggers a resume


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def get_total(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=CONNECT_TIMEOUT)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except requests.RequestException:
        return None


def download(url: str, out: str, max_retries: int = 0) -> bool:
    session = requests.Session()
    session.headers["User-Agent"] = UA
    # identity: never let the server gzip the stream (would break byte-offset
    # accounting for range resume, and a .zip is already compressed anyway).
    session.headers["Accept-Encoding"] = "identity"

    total = get_total(session, url)
    if total:
        print(f"Total size : {human(total)} ({total:,} bytes)")
    else:
        print("Total size : unknown (HEAD gave no Content-Length)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"Saving to  : {out}\n")

    attempt = 0
    while True:
        done = os.path.getsize(out) if os.path.exists(out) else 0
        if total and done >= total:
            break

        headers = {"Range": f"bytes={done}-"} if done else {}
        if done:
            pct = f" ({100 * done / total:.1f}%)" if total else ""
            print(f"[resume] continuing from {human(done)}{pct}")
        try:
            with session.get(url, headers=headers, stream=True,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code == 416:
                    print("Range not satisfiable (416) - file is already complete.")
                    break
                if done and r.status_code == 200:
                    # This server *intermittently* ignores Range and offers the
                    # whole file from byte 0. NEVER truncate our partial progress
                    # over that - drop this response and retry the ranged request
                    # (the next attempt almost always returns 206).
                    attempt += 1
                    wait = min(30, 2 ** min(attempt, 5))
                    print(f"\n[!] Server ignored Range at {human(done)} (HTTP 200); "
                          f"keeping progress, retrying in {wait}s (attempt {attempt}).")
                    time.sleep(wait)
                    continue
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                mode = "ab" if done else "wb"

                last_t = time.time()
                last_b = done
                with open(out, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last_t >= 1.0:
                            speed = (done - last_b) / (now - last_t)
                            if total:
                                pct = f"{100 * done / total:.1f}%"
                                eta = human_time((total - done) / speed) if speed > 0 else "?"
                                bar = f"{human(done)}/{human(total)} ({pct})"
                            else:
                                pct, eta, bar = "?", "?", human(done)
                            sys.stdout.write(
                                f"\r  {bar}  {human(speed)}/s  ETA {eta}      ")
                            sys.stdout.flush()
                            last_t, last_b = now, done
            # Connection closed cleanly; loop re-checks completion at the top.
            attempt = 0
        except (requests.RequestException, OSError) as exc:
            attempt += 1
            have = os.path.getsize(out) if os.path.exists(out) else 0
            print(f"\n[!] Interrupted at {human(have)} - {type(exc).__name__}: {exc}")
            if max_retries and attempt >= max_retries:
                print("Reached max retries; stopping. Re-run to resume.")
                return False
            wait = min(30, 2 ** min(attempt, 5))
            print(f"    Retrying in {wait}s (attempt {attempt})...")
            time.sleep(wait)

    final = os.path.getsize(out)
    print(f"\n\n[OK] Download complete: {human(final)} ({final:,} bytes) -> {out}")
    if total and final != total:
        print(f"[!] Size mismatch! expected {total:,}, got {final:,}. Re-run to fix.")
        return False
    print("   Size matches the server's Content-Length. "
          "You'll need the brainclinics password to unzip the _Encr.zip.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL, help="download URL")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output path")
    ap.add_argument("--max-retries", type=int, default=0,
                    help="0 = retry forever (default); N = give up after N failures")
    args = ap.parse_args()
    try:
        ok = download(args.url, args.out, args.max_retries)
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume from where it left off.")
        return 130
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
