"""Submit five concurrent jobs and record observed end-to-end latencies."""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


async def run(base_url: str, timeout: float) -> dict:
    queries = [
        "Investigate INC-104 and calculate the p95 latency increase",
        "Investigate INC-207 and calculate the p95 latency increase",
        "Investigate INC-315 and calculate the p95 latency increase",
        "Investigate INC-421 and calculate the p95 latency increase",
        "Investigate INC-999 and calculate the p95 latency increase",
    ]
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        async def submit_and_poll(query: str) -> dict:
            start = time.perf_counter()
            response = await client.post("/tasks", json={"query": query})
            response.raise_for_status()
            job_id = response.json()["job_id"]
            deadline = start + timeout
            while time.perf_counter() < deadline:
                try:
                    current = await client.get(f"/tasks/{job_id}")
                except httpx.TransportError:
                    await asyncio.sleep(1)
                    continue
                current.raise_for_status()
                job = current.json()
                if job["status"] in {"DONE", "FAILED", "WAITING_APPROVAL"}:
                    return {"job_id": job_id, "query": query, "status": job["status"], "answer": job.get("answer"), "seconds": round(time.perf_counter() - start, 3)}
                await asyncio.sleep(1)
            return {"job_id": job_id, "status": "TIMEOUT", "seconds": round(time.perf_counter() - start, 3)}

        jobs = await asyncio.gather(*(submit_and_poll(query) for query in queries))
    completed = sorted(job["seconds"] for job in jobs if job["status"] == "DONE")
    p95 = completed[math.ceil(0.95 * len(completed)) - 1] if len(completed) == 5 else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
        "p95_seconds_nearest_rank": p95,
        "note": "Client-observed p95 is not a Phoenix trace metric. Verify all five job IDs in Phoenix before claiming trace evidence.",
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run five concurrent API jobs and measure latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path; never fabricate a dashboard screenshot.")
    args = parser.parse_args()
    result = await run(args.base_url, args.timeout)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
