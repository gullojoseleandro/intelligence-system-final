import asyncio
import os
import unittest

import fakeredis.aioredis
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.main import TaskAccepted, TaskView
from pydantic import ValidationError


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def test_response_models_reject_invalid_status_and_approval(self) -> None:
        with self.assertRaises(ValidationError):
            TaskAccepted(job_id="abc", status="BROKEN")
        with self.assertRaises(ValidationError):
            TaskView(job_id="abc", status="WAITING_APPROVAL", approval={"unexpected": "value"})

    async def test_request_rejects_blank_query(self) -> None:
        response = await self.client.post("/tasks", json={"query": "   "})
        self.assertEqual(response.status_code, 422)

    async def asyncSetUp(self) -> None:
        self.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        self.app = create_app(redis_client=self.redis, admin_token="test-admin-token")
        self.lifespan = LifespanManager(self.app)
        await self.lifespan.__aenter__()
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        await self.redis.aclose()

    async def test_enqueue_and_poll_are_nonblocking(self) -> None:
        response = await self.client.post("/tasks", json={"query": "Investigate INC-104"})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        self.assertEqual(response.json()["status"], "PENDING")
        self.assertEqual((await self.client.get(f"/tasks/{job_id}")).json()["status"], "PENDING")
        self.assertEqual(await self.redis.llen("agent:queue"), 1)
        self.assertEqual((await self.client.get("/tasks/not-found")).status_code, 404)

    async def test_five_concurrent_requests_get_unique_ids(self) -> None:
        responses = await asyncio.gather(*[
            self.client.post("/tasks", json={"query": f"Investigate INC-{100 + index}"})
            for index in range(5)
        ])
        self.assertTrue(all(response.status_code == 202 for response in responses))
        self.assertEqual(len({response.json()["job_id"] for response in responses}), 5)
        self.assertEqual(await self.redis.llen("agent:queue"), 5)

    async def test_approval_requires_token_and_is_single_use(self) -> None:
        response = await self.client.post("/tasks", json={"query": "Deploy fix for INC-104"})
        job_id = response.json()["job_id"]
        await self.redis.hset(f"agent:job:{job_id}", mapping={"status": "WAITING_APPROVAL"})

        denied = await self.client.post(f"/tasks/{job_id}/approve", json={"approved": True})
        self.assertEqual(denied.status_code, 401)
        headers = {"X-Admin-Token": "test-admin-token"}
        accepted = await self.client.post(f"/tasks/{job_id}/approve", json={"approved": True}, headers=headers)
        self.assertEqual(accepted.status_code, 202)
        self.assertEqual((await self.client.get(f"/tasks/{job_id}")).json()["status"], "PENDING")
        repeated = await self.client.post(f"/tasks/{job_id}/approve", json={"approved": True}, headers=headers)
        self.assertEqual(repeated.status_code, 409)

    async def test_competing_approvals_enqueue_once(self) -> None:
        response = await self.client.post("/tasks", json={"query": "Deploy fix for INC-104"})
        job_id = response.json()["job_id"]
        await self.redis.lpop("agent:queue")
        await self.redis.hset(f"agent:job:{job_id}", mapping={"status": "WAITING_APPROVAL"})
        headers = {"X-Admin-Token": "test-admin-token"}

        responses = await asyncio.gather(*[
            self.client.post(f"/tasks/{job_id}/approve", json={"approved": value}, headers=headers)
            for value in (True, False)
        ])
        self.assertEqual(sorted(response.status_code for response in responses), [202, 409])
        self.assertEqual(await self.redis.llen("agent:queue"), 1)
