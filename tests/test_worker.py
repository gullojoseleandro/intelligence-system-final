import unittest
from unittest.mock import patch
from types import SimpleNamespace

import fakeredis.aioredis

from app.jobs import JobStore
from app.worker import Worker, create_router


class FakeGraph:
    def __init__(self, *, failure: bool = False, interrupted: bool = False) -> None:
        self.failure = failure
        self.interrupted = interrupted
        self.inputs = []

    async def ainvoke(self, value, config):
        self.inputs.append(value)
        if self.failure:
            raise RuntimeError("model failed")
        if self.interrupted:
            return {"__interrupt__": [{"value": {"reason": "critical operation"}}]}
        return {"answer": "Verified response"}

    async def aget_state(self, config):
        return SimpleNamespace(values={"query": "Recovered query"})


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        self.store = JobStore(self.redis)

    async def asyncTearDown(self) -> None:
        await self.redis.aclose()

    async def test_success_and_failure_are_persisted(self) -> None:
        success = await self.store.create("Investigate INC-104")
        await Worker(self.store, FakeGraph()).process_once(timeout=1)
        self.assertEqual((await self.store.get(success))["status"], "DONE")

        failure = await self.store.create("Investigate INC-207")
        await Worker(self.store, FakeGraph(failure=True)).process_once(timeout=1)
        job = await self.store.get(failure)
        self.assertEqual(job["status"], "FAILED")
        self.assertIn("RuntimeError", job["error"])
        self.assertNotIn("model failed", job["error"])

    async def test_interrupt_waits_for_approval(self) -> None:
        job_id = await self.store.create("Deploy fix for INC-104")
        await Worker(self.store, FakeGraph(interrupted=True)).process_once(timeout=1)
        self.assertEqual((await self.store.get(job_id))["status"], "WAITING_APPROVAL")
        self.assertEqual(await self.redis.llen("agent:processing"), 0)

    async def test_recovered_job_continues_saved_checkpoint(self) -> None:
        job_id = await self.store.create("Investigate INC-104")
        self.assertEqual(await self.store.claim(timeout=1), job_id)
        await self.store.update(job_id, "RUNNING")
        self.assertEqual(await self.store.requeue_interrupted(), 1)

        graph = FakeGraph()
        await Worker(self.store, graph).process_once(timeout=1)
        self.assertIsNone(graph.inputs[0])
        self.assertEqual((await self.store.get(job_id))["status"], "DONE")


class WorkerProviderTests(unittest.TestCase):
    def test_nvidia_mode_selects_nvidia_supervisor(self) -> None:
        with patch("app.worker.NvidiaSupervisor") as supervisor:
            self.assertIs(create_router("nvidia"), supervisor.return_value)
        supervisor.assert_called_once_with("meta/llama-3.1-8b-instruct")

    def test_existing_offline_mode_stays_offline(self) -> None:
        self.assertEqual(type(create_router("offline")).__name__, "OfflineDemoRouter")
