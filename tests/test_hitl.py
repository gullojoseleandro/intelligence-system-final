import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph, initial_state
from app.demo import OfflineDemoRouter


class HitlTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_query_does_not_pause(self) -> None:
        graph = build_graph(OfflineDemoRouter(), InMemorySaver())
        config = {"configurable": {"thread_id": "normal-1"}, "recursion_limit": 16}
        finished = await graph.ainvoke(initial_state("Investigate INC-104"), config)
        self.assertNotIn("__interrupt__", finished)
        self.assertIn("INC-104", finished["answer"])
        self.assertFalse(finished["critical"])

    async def test_critical_query_pauses_and_resumes(self) -> None:
        graph = build_graph(OfflineDemoRouter(), InMemorySaver())
        config = {"configurable": {"thread_id": "critical-1"}, "recursion_limit": 16}
        paused = await graph.ainvoke(initial_state("Deploy fix for INC-104"), config)
        self.assertIn("__interrupt__", paused)
        self.assertEqual(paused["__interrupt__"][0].value["reason"], "critical operation")

        finished = await graph.ainvoke(Command(resume={"approved": True}), config)
        self.assertIn("INC-104", finished["answer"])
        self.assertTrue(finished["approval_granted"])

    async def test_rejection_stops_without_specialist_calls(self) -> None:
        graph = build_graph(OfflineDemoRouter(), InMemorySaver())
        config = {"configurable": {"thread_id": "critical-2"}, "recursion_limit": 16}
        await graph.ainvoke(initial_state("Delete INC-104"), config)
        finished = await graph.ainvoke(Command(resume={"approved": False}), config)
        self.assertIn("declined", finished["answer"].lower())
        self.assertEqual(finished["contributions"], [])
