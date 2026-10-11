"""Integration tests that need PyBullet (skipped if it is not installed)."""

import importlib.util
import unittest

HAVE_PYBULLET = importlib.util.find_spec("pybullet") is not None


class TestSceneRandomizer(unittest.TestCase):
    def test_deterministic_and_complete(self):
        from fetchbot.world import SceneRandomizer

        a = SceneRandomizer(seed=7).randomize()
        b = SceneRandomizer(seed=7).randomize()
        self.assertEqual(a.robot_start, b.robot_start)
        self.assertEqual({z.name for z in a.zones}, {"dock", "table", "kitchen"})
        self.assertEqual(len(a.objects), 4)
        self.assertNotEqual(a.robot_start, SceneRandomizer(seed=8).randomize().robot_start)


@unittest.skipUnless(HAVE_PYBULLET, "pybullet not installed")
class TestEpisode(unittest.TestCase):
    def test_fetch_episode_succeeds(self):
        from fetchbot.evaluation.episode import EpisodeConfig, run_episode

        r = run_episode(
            EpisodeConfig(
                command="bring the red box to the dock", seed=2, parser="rule", max_sim_time=120
            )
        )
        self.assertTrue(r.success, r.failure)
        self.assertGreater(r.distance_m, 0.5)
        self.assertEqual(r.grabbed, "obj_0_red_box")

    def test_multi_step_program_runs_in_order(self):
        from fetchbot.evaluation.episode import EpisodeConfig, run_episode

        r = run_episode(
            EpisodeConfig(
                command="bring the red ball to the dock, then bring the green box to the table",
                condition="shapes",
                seed=3,
                parser="rule",
                with_person=False,
                max_sim_time=300,
            )
        )
        self.assertTrue(r.success, r.failure)
        self.assertEqual((r.steps, r.steps_ok), (2, 2))
        self.assertEqual(
            [s["delivered"] for s in r.step_log], [["obj_1_red_ball"], ["obj_3_green_box"]]
        )


if __name__ == "__main__":
    unittest.main()
