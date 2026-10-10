"""Synthesise one system's jobs from a job list, optionally sharded (resumable).

Usage (on a worker, from the bundle root, inside the system's venv):
    python -m condsites.synthesize --jobs jobs/pathway.jsonl --system qwen17 \
        --shard 0 --num-shards 6 --output-dir audio

Writes <output-dir>/<system>/<job_id>.wav and appends one JSON line per job to
<output-dir>/<system>/status-shard<k>.jsonl. Existing WAVs are skipped.
Full-condition prompts are built at 24 kHz (the July run used 16 kHz).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from pathlib import Path

import soundfile as sf

from condsites.backends.base import Condition
from condsites.references import RefItem
from condsites.systems import create_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="stop after N jobs (smoke tests)")
    args = parser.parse_args()

    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines()]
    jobs = [job for job in jobs if job["system"] == args.system]
    jobs = [job for index, job in enumerate(sorted(jobs, key=lambda job: job["job_id"])) if index % args.num_shards == args.shard]
    output_dir = args.output_dir / args.system
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"status-shard{args.shard}.jsonl"
    pending = [job for job in jobs if not (output_dir / f"{job['job_id']}.wav").exists()]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{args.system} shard {args.shard}/{args.num_shards}: {len(jobs)} jobs, {len(pending)} pending", flush=True)

    backend = create_backend(args.system)
    with status_path.open("a", encoding="utf-8") as status:
        for index, job in enumerate(pending):
            started = time.time()
            record = {"job_id": job["job_id"], "system": args.system}
            try:
                refs = tuple(
                    RefItem(id=ref["id"], speaker_id=job["speaker_id"], text=ref["text"], path=ref["path"],
                            sample_rate=24000, duration=ref["duration"] or sf.info(ref["path"]).duration)
                    for ref in job["refs"]
                )
                with tempfile.TemporaryDirectory() as work_dir:
                    if job["condition"].startswith("site:"):
                        prepared = backend.prepare_site(refs, job["condition"], Path(work_dir))
                    else:
                        prepared = backend.prepare_reference(refs, Condition(job["condition"]), Path(work_dir))
                    result = backend.synthesize(prepared, job["target_text"], job["seed"])
                wav_path = output_dir / f"{job['job_id']}.wav"
                temporary = wav_path.with_suffix(".tmp.wav")
                sf.write(temporary, result.waveform, result.sample_rate)
                temporary.rename(wav_path)
                record.update(status="success", duration_s=round(len(result.waveform) / result.sample_rate, 3))
            except Exception as error:  # record and continue; failures are analysed, not hidden
                record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-2000:])
            record["runtime_s"] = round(time.time() - started, 3)
            status.write(json.dumps(record) + "\n")
            status.flush()
            if (index + 1) % 50 == 0:
                print(f"{index + 1}/{len(pending)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
