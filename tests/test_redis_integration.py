import os
import unittest
from uuid import uuid4

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command

from app.demo import OfflineDemoRouter
from app.graph import build_graph, initial_state


@unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "Set TEST_REDIS_URL for a Redis 8 integration check")
class RedisCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_survives_graph_reconstruction(self) -> None:
        url = os.environ["TEST_REDIS_URL"]
        config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 16}
        async with AsyncRedisSaver.from_conn_string(url) as saver:
            graph = build_graph(OfflineDemoRouter(), saver)
            paused = await graph.ainvoke(initial_state("Deploy fix for INC-104"), config)
            self.assertIn("__interrupt__", paused)

        async with AsyncRedisSaver.from_conn_string(url) as saver:
            graph = build_graph(OfflineDemoRouter(), saver)
            finished = await graph.ainvoke(Command(resume={"approved": True}), config)
            self.assertIn("INC-104", finished["answer"])
            self.assertTrue(finished["approval_granted"])
