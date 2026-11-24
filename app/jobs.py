import json
from datetime import datetime, timezone
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import WatchError


QUEUE_KEY = "agent:queue"
PROCESSING_KEY = "agent:processing"


def job_key(job_id: str) -> str:
    return f"agent:job:{job_id}"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def create(self, query: str) -> str:
        job_id = str(uuid4())
        stamp = now_utc()
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(job_key(job_id), mapping={
                "job_id": job_id, "query": query, "status": "PENDING",
                "created_at": stamp, "updated_at": stamp,
            })
            pipeline.lpush(QUEUE_KEY, job_id)
            await pipeline.execute()
        return job_id

    async def get(self, job_id: str) -> dict[str, str] | None:
        job = await self.redis.hgetall(job_key(job_id))
        return job or None

    async def claim(self, timeout: int = 1) -> str | None:
        return await self.redis.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout=timeout)

    async def acknowledge(self, job_id: str) -> None:
        await self.redis.lrem(PROCESSING_KEY, 1, job_id)

    async def update(self, job_id: str, status: str, **fields: str) -> None:
        await self.redis.hset(job_key(job_id), mapping={"status": status, "updated_at": now_utc(), **fields})

    async def approve(self, job_id: str, approved: bool) -> str:
        key = job_key(job_id)
        for _ in range(3):
            async with self.redis.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(key)
                    status = await pipeline.hget(key, "status")
                    if status is None:
                        return "missing"
                    if status != "WAITING_APPROVAL":
                        return "invalid"
                    pipeline.multi()
                    pipeline.hset(key, mapping={
                        "status": "PENDING", "approval_decision": json.dumps(approved), "updated_at": now_utc(),
                    })
                    pipeline.lpush(QUEUE_KEY, job_id)
                    await pipeline.execute()
                    return "accepted"
                except WatchError:
                    continue
        return "invalid"

    async def requeue_interrupted(self) -> int:
        """Recover a job claimed by this single worker before its previous process exited."""
        recovered = 0
        for job_id in await self.redis.lrange(PROCESSING_KEY, 0, -1):
            job = await self.get(job_id)
            async with self.redis.pipeline(transaction=True) as pipeline:
                if job and job["status"] in {"PENDING", "RUNNING"}:
                    pipeline.hset(job_key(job_id), mapping={
                        "status": "PENDING", "resume_unfinished": "true", "updated_at": now_utc(),
                    })
                    pipeline.lpush(QUEUE_KEY, job_id)
                    recovered += 1
                pipeline.lrem(PROCESSING_KEY, 1, job_id)
                await pipeline.execute()
        return recovered
