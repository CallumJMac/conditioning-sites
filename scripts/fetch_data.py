"""Download the evaluation data and verify it byte-for-byte.

LibriTTS-R: the 1,731 test-clean utterances listed in
data/libritts_r_aligned/manifest_40spk.json, taken from the Hugging Face parquet
export of mythicinfinity/libritts_r (config "clean", split "test.clean") and
written as WAV exactly as in the original runs (decoded, then soundfile.write
with default settings). Each file is checked against
data/libritts_r_aligned/SHA256SUMS.

Seed-TTS-eval test-en: the official test-set archive from the seed-tts-eval
repository (Google Drive), checked against its SHA-256, then the English part is
extracted to data/seedtts_en/.

Usage:
    python scripts/fetch_data.py                 # both datasets
    python scripts/fetch_data.py --only libritts --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRITTS_DIR = ROOT / "data" / "libritts_r_aligned"
SEEDTTS_DIR = ROOT / "data" / "seedtts_en"
PARQUET_INDEX = "https://huggingface.co/api/datasets/mythicinfinity/libritts_r/parquet/clean/test.clean"
SEEDTTS_DRIVE_ID = "1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
SEEDTTS_SHA256 = "65b49df7a5f4714042f9849fa216326f0d77dfe37496878b9b2240a0d2a2f4f9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_libritts(limit: int | None) -> int:
    import pyarrow.parquet as pq
    import soundfile as sf

    expected = dict(reversed(line.split("  ", 1)) for line in (LIBRITTS_DIR / "SHA256SUMS").read_text().splitlines())
    wanted = {Path(path).stem: path for path in expected}
    if limit:
        wanted = dict(list(wanted.items())[:limit])
    shards = json.loads(urllib.request.urlopen(PARQUET_INDEX, timeout=60).read())
    written = mismatched = 0
    for shard in shards:
        if written + mismatched == len(wanted):
            break
        with tempfile.NamedTemporaryFile(suffix=".parquet") as local:
            print(f"downloading {shard}", flush=True)
            urllib.request.urlretrieve(shard, local.name)
            table = pq.read_table(local.name, columns=["id", "audio"])
        for uid, audio in zip(table.column("id").to_pylist(), table.column("audio").to_pylist(), strict=True):
            if uid not in wanted:
                continue
            path = LIBRITTS_DIR / wanted[uid]
            path.parent.mkdir(parents=True, exist_ok=True)
            waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]))
            sf.write(path, waveform, sample_rate)
            if sha256(path) == expected[wanted[uid]]:
                written += 1
            else:
                mismatched += 1
                print(f"checksum mismatch: {wanted[uid]}", file=sys.stderr)
    missing = len(wanted) - written - mismatched
    print(f"LibriTTS-R: {written} verified, {mismatched} mismatched, {missing} not found")
    return 0 if mismatched == missing == 0 else 1


def fetch_seedtts() -> int:
    archive = ROOT / "data" / "seedtts_testset.tar"
    if not archive.exists():
        subprocess.run([sys.executable, "-m", "gdown", SEEDTTS_DRIVE_ID, "-O", str(archive)], check=True)
    if sha256(archive) != SEEDTTS_SHA256:
        print("Seed-TTS-eval archive checksum mismatch", file=sys.stderr)
        return 1
    with tarfile.open(archive) as handle:
        members = [member for member in handle.getmembers() if member.name.startswith("seedtts_testset/en/")]
        for member in members:
            member.name = member.name.removeprefix("seedtts_testset/en/")
        SEEDTTS_DIR.mkdir(parents=True, exist_ok=True)
        handle.extractall(SEEDTTS_DIR, members=members, filter="data")
    print(f"Seed-TTS-eval test-en: {len(members)} files extracted to {SEEDTTS_DIR.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("libritts", "seedtts"))
    parser.add_argument("--limit", type=int, help="LibriTTS-R only: stop after this many files (for a quick check)")
    args = parser.parse_args()
    status = 0
    if args.only in (None, "libritts"):
        status |= fetch_libritts(args.limit)
    if args.only in (None, "seedtts"):
        status |= fetch_seedtts()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
