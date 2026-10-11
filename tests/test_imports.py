"""Smoke test: every subpackage imports and the class hierarchy is wired up."""

import importlib
import unittest

SUBPACKAGES = [
    "robots",
    "data",
    "perception",
    "mapping",
    "navigation",
    "tasks",
    "language",
    "world",
    "evaluation",
    "ui",
    "export",
]


class TestImports(unittest.TestCase):
    def test_subpackages_import(self):
        for name in SUBPACKAGES:
            with self.subTest(name=name):
                importlib.import_module(f"fetchbot.{name}")

    def test_inheritance(self):
        from fetchbot.data import CameraStream, LidarStream, OdomStream, SensorStream
        from fetchbot.export import Exporter, MapExporter, ReportExporter, VideoExporter
        from fetchbot.language import CommandParser, LLMParser, RuleBasedParser
        from fetchbot.navigation import GridNavigator, Navigator
        from fetchbot.perception import ColorDetector, Detector, OpenVocabDetector
        from fetchbot.robots import HuskyRobot, Robot
        from fetchbot.tasks import (
            ApproachState,
            DeliverState,
            DoneState,
            ExploreState,
            GrabState,
            ReleaseState,
            State,
        )
        from fetchbot.ui import CliInterface, GuiInterface, Interface

        hierarchy = {
            Robot: [HuskyRobot],
            SensorStream: [CameraStream, LidarStream, OdomStream],
            Detector: [OpenVocabDetector, ColorDetector],
            Navigator: [GridNavigator],
            State: [ExploreState, ApproachState, GrabState, DeliverState, ReleaseState, DoneState],
            CommandParser: [RuleBasedParser, LLMParser],
            Interface: [CliInterface, GuiInterface],
            Exporter: [VideoExporter, ReportExporter, MapExporter],
        }
        for base, subs in hierarchy.items():
            for sub in subs:
                with self.subTest(base=base.__name__, sub=sub.__name__):
                    self.assertTrue(issubclass(sub, base))

    def test_task_manager_default_states(self):
        from fetchbot.tasks import TaskManager

        tm = TaskManager()
        self.assertEqual(tm.current, "explore")
        self.assertEqual(
            list(tm.states), ["explore", "approach", "grab", "deliver", "release", "done"]
        )


if __name__ == "__main__":
    unittest.main()
