import unittest
from unittest.mock import patch

import httpx

from load_test import run


class FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.submitted = 0
        self.polls: dict[str, int] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, path: str, json: dict) -> httpx.Response:
        job_id = f"job-{self.submitted}"
        self.submitted += 1
        return httpx.Response(202, json={"job_id": job_id}, request=httpx.Request("POST", f"http://test{path}"))

    async def get(self, path: str) -> httpx.Response:
        job_id = path.rsplit("/", 1)[-1]
        self.polls[job_id] = self.polls.get(job_id, 0) + 1
        if job_id == "job-0" and self.polls[job_id] == 1:
            raise httpx.ReadTimeout("temporary overload")
        return httpx.Response(200, json={"status": "DONE", "answer": "grounded"}, request=httpx.Request("GET", f"http://test{path}"))


class LoadTestTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_poll_error_does_not_abort_five_jobs(self) -> None:
        with patch("load_test.httpx.AsyncClient", FakeClient):
            result = await run("http://test", timeout=5)
        self.assertEqual(len(result["jobs"]), 5)
        self.assertTrue(all(job["status"] == "DONE" for job in result["jobs"]))
        self.assertTrue(all(job["answer"] == "grounded" for job in result["jobs"]))
        self.assertIsNotNone(result["p95_seconds_nearest_rank"])
