import asyncio
import json
import logging
import os

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from redis.asyncio import Redis

from app.demo import OfflineDemoRouter
from app.graph import build_graph, initial_state
from app.jobs import JobStore
from app.observability import load_settings
from app.supervisor import NvidiaSupervisor, OpenAISupervisor, Router


logger = logging.getLogger(__name__)


def create_router(mode: str) -> Router:
    if mode == "offline":
        return OfflineDemoRouter()
    if mode == "nvidia":
        return NvidiaSupervisor(os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct"))
    return OpenAISupervisor(os.getenv("MODEL_NAME", "gpt-4o-mini"))


class Worker:
    def __init__(self, store: JobStore, graph) -> None:
        self.store = store
        self.graph = graph

    async def process_once(self, timeout: int = 1) -> bool:
        job_id = await self.store.claim(timeout=timeout)
        if job_id is None:
            return False
        try:
            job = await self.store.get(job_id)
            if not job or job["status"] != "PENDING":
                return True
            await self.store.update(job_id, "RUNNING")
            config = {"configurable": {"thread_id": job_id}, "metadata": {"job_id": job_id}, "recursion_limit": 16}
            if "approval_decision" in job:
                value = Command(resume={"approved": json.loads(job["approval_decision"])})
            elif job.get("resume_unfinished") == "true":
                snapshot = await self.graph.aget_state(config)
                value = None if snapshot.values else initial_state(job["query"])
            else:
                value = initial_state(job["query"])
            result = await self.graph.ainvoke(value, config)
            interruptions = result.get("__interrupt__", [])
            if interruptions:
                item = interruptions[0]
                payload = item.get("value", {}) if isinstance(item, dict) else item.value
                await self.store.update(job_id, "WAITING_APPROVAL", approval=json.dumps(payload))
            else:
                await self.store.update(job_id, "DONE", answer=result["answer"])
            return True
        except asyncio.CancelledError:
            await self.store.update(job_id, "PENDING", resume_unfinished="true")
            await self.store.redis.lpush("agent:queue", job_id)
            raise
        except Exception as error:
            logger.exception("Job %s failed", job_id)
            await self.store.update(job_id, "FAILED", error=f"{type(error).__name__}: Task execution failed")
            return True
        finally:
            await self.store.acknowledge(job_id)

    async def run_forever(self) -> None:
        while True:
            await self.process_once(timeout=1)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mode, redis_url = load_settings()
    redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        await redis.ping()
        async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
            router = create_router(mode)
            graph = build_graph(router, saver)
            store = JobStore(redis)
            count = await store.requeue_interrupted()
            logger.info("Worker started; recovered %d interrupted jobs", count)
            await Worker(store, graph).run_forever()
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
