"""A*, costmap and occupancy-grid unit tests (pure numpy, no simulator)."""

import math
import unittest

import numpy as np

from fetchbot.data import LaserScan
from fetchbot.language.schema import FetchStep
from fetchbot.mapping import FrontierExplorer, OccupancyGrid, SemanticMap
from fetchbot.navigation import astar, build_costmap
from fetchbot.tasks.program import Geometry, ObjInfo, needs_survey, select_candidates
from fetchbot.types import Detection, Pose2D


class TestAStar(unittest.TestCase):
    def test_straight_line(self):
        cost = np.ones((5, 10))
        path = astar(cost, (2, 0), (2, 9))
        self.assertEqual(path[0], (2, 0))
        self.assertEqual(path[-1], (2, 9))
        self.assertEqual(len(path), 10)

    def test_detours_around_wall(self):
        cost = np.ones((10, 10))
        cost[0:8, 5] = np.inf  # wall with a gap at the bottom rows
        path = astar(cost, (0, 0), (0, 9))
        self.assertIsNotNone(path)
        self.assertTrue(all(np.isfinite(cost[c]) for c in path))
        self.assertTrue(any(r >= 8 for r, c in path if c == 5))

    def test_unreachable(self):
        cost = np.ones((6, 6))
        cost[:, 3] = np.inf
        self.assertIsNone(astar(cost, (0, 0), (0, 5)))

    def test_no_corner_cutting(self):
        cost = np.ones((3, 3))
        cost[0, 1] = cost[1, 0] = np.inf
        self.assertIsNone(astar(cost, (0, 0), (1, 1)))

    def test_goal_radius_reaches_goal_inside_obstacle(self):
        cost = np.ones((10, 10))
        cost[4:7, 4:7] = np.inf  # goal (5, 5) is inside an obstacle
        self.assertIsNone(astar(cost, (0, 0), (5, 5)))
        path = astar(cost, (0, 0), (5, 5), goal_radius=2.0)
        self.assertIsNotNone(path)
        r, c = path[-1]
        self.assertLessEqual(math.hypot(r - 5, c - 5), 2.0)

    def test_costmap_layers(self):
        dist = np.array([[0.0, 1.0, 2.5, 5.0, 9.0]])
        unknown = np.array([[False, False, False, False, True]])
        cost = build_costmap(dist, unknown, lethal_cells=2.0, inflation_cells=6.0)
        self.assertTrue(np.isinf(cost[0, 0]) and np.isinf(cost[0, 1]))
        self.assertGreater(cost[0, 2], cost[0, 3])  # cost decays away from obstacles
        self.assertAlmostEqual(cost[0, 4], 2.0)  # unknown penalty


def _scan(pose, ranges, max_range=5.0):
    angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    return LaserScan(pose, angles, np.asarray(ranges, dtype=float), max_range, 0.0)


class TestOccupancyGrid(unittest.TestCase):
    def test_coordinates_roundtrip(self):
        g = OccupancyGrid(4.0, 3.0, resolution=0.1, origin=(-0.5, -0.5))
        r, c = g.world_to_cell(1.23, 0.47)
        x, y = g.cell_to_world(r, c)
        self.assertAlmostEqual(x, 1.25)
        self.assertAlmostEqual(y, 0.45)

    def test_scan_marks_free_and_occupied(self):
        g = OccupancyGrid(6.0, 6.0, resolution=0.1)
        pose = Pose2D(3.0, 3.0)
        for _ in range(3):  # a ring wall 2 m away
            g.update_from_scan(_scan(pose, [2.0] * 360))
        self.assertTrue(g.free[g.world_to_cell(3.0, 3.0)])
        self.assertTrue(g.free[g.world_to_cell(4.0, 3.0)])
        self.assertTrue(g.occupied[g.world_to_cell(5.0, 3.0)])
        self.assertTrue(g.unknown[g.world_to_cell(5.6, 3.0)])  # behind the wall
        self.assertGreater(g.known_fraction(), 0.2)

    def test_obstacle_distance(self):
        g = OccupancyGrid(2.0, 2.0, resolution=0.1)
        g.logodds[10, 10] = 3.0
        d = g.obstacle_distance(5.0)
        self.assertEqual(d[10, 10], 0.0)
        self.assertAlmostEqual(float(d[10, 13]), 3.0)
        self.assertEqual(d[0, 0], 5.0)

    def test_transient_obstacles(self):
        g = OccupancyGrid(6.0, 6.0, resolution=0.1)
        pose = Pose2D(3.0, 3.0)
        for _ in range(3):  # ring wall 2 m away; the cell 1 m west is seen free
            g.update_from_scan(_scan(pose, [2.0] * 360))
        for _ in range(8):  # then something steps into that cell
            g.update_from_scan(_scan(pose, [1.0] + [2.0] * 359))
        person, wall = g.world_to_cell(2.0, 3.0), g.world_to_cell(5.0, 3.0)
        self.assertTrue(g.occupied[person] and g.transient[person])
        self.assertTrue(g.occupied[wall] and not g.transient[wall])
        self.assertEqual(g.obstacle_distance(5.0, ignore_transient=True)[wall], 0.0)
        self.assertGreater(g.obstacle_distance(5.0, ignore_transient=True)[person], 0.0)

    def test_frontier_goal(self):
        g = OccupancyGrid(6.0, 6.0, resolution=0.1)
        pose = Pose2D(3.0, 3.0)
        # Beams hit a wall 2 m away except an open gap (misses) toward +x.
        ranges = [2.0] * 360
        for i in range(170, 190):
            ranges[i] = np.inf
        for _ in range(3):
            g.update_from_scan(_scan(pose, ranges, max_range=2.5))
        ex = FrontierExplorer()
        ex.update(g)
        goal = ex.next_goal(pose)
        self.assertIsNotNone(goal)
        self.assertGreater(goal.x, 4.5)  # the frontier is in the open gap


class TestSemanticMap(unittest.TestCase):
    def test_fuses_nearby_detections(self):
        sm = SemanticMap(fuse_radius=0.5)
        det = Detection("red box", 0.9, (0, 0, 1, 1), Pose2D(1.0, 1.0))
        a = sm.add_detection(det)
        b = sm.add_detection(Detection("red box", 0.8, (0, 0, 1, 1), Pose2D(1.2, 1.0)))
        c = sm.add_detection(Detection("red box", 0.8, (0, 0, 1, 1), Pose2D(4.0, 4.0)))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertAlmostEqual(sm.get(a).map_pose.x, 1.1)
        self.assertEqual(sm.find_ids("red box", min_sightings=2), [a])
        sm.remove(a)
        self.assertEqual(len(sm), 1)


class TestTargetSelection(unittest.TestCase):
    """Program target selection (shared by the robot's belief and ground-truth scoring)."""

    objs = [
        ObjInfo("rb", "red", "box", 1.0, 1.0),
        ObjInfo("rl", "red", "ball", 8.0, 1.0),
        ObjInfo("gb", "green", "box", 5.0, 6.0),
        ObjInfo("bc", "blue", "cylinder", 9.0, 6.5),
    ]
    geo = Geometry(
        zones={"dock": (9.0, 7.0, 0.8), "table": (1.0, 7.0, 0.8)},
        shelves=[(4.5, 5.5, 5.5, 5.8)],
        regions={"west room": (0, 0, 5, 8), "east room": (5, 0, 10, 8)},
        robot_xy=(0.0, 0.0),
    )

    def ids(self, **kw):
        kw.setdefault("destination", "table")
        return [o.oid for o in select_candidates(FetchStep(**kw), self.objs, self.geo)]

    def test_color_and_shape(self):
        self.assertEqual(self.ids(color="red", shape="ball"), ["rl"])
        self.assertEqual(self.ids(color="red"), ["rb", "rl"])  # nearest to the robot first
        self.assertEqual(self.ids(shape="box", quantifier="all"), ["rb", "gb"])

    def test_objects_already_at_destination_are_skipped(self):
        self.assertEqual(self.ids(color="blue", destination="dock"), [])

    def test_relations(self):
        self.assertEqual(self.ids(shape="box", relation="nearest_to", anchor="shelf"), ["gb"])
        self.assertEqual(self.ids(shape="box", relation="farthest_from", anchor="robot"), ["gb"])
        self.assertEqual(self.ids(color="red", relation="rightmost"), ["rl"])
        self.assertEqual(self.ids(shape="box", relation="northmost"), ["gb"])
        self.assertEqual(self.ids(color="red", relation="in_room", anchor="east room"), ["rl"])
        self.assertEqual(
            self.ids(shape="box", relation="nearest_to", anchor="blue cylinder"), ["gb"]
        )

    def test_unknown_anchor_raises(self):
        with self.assertRaises(LookupError):
            self.ids(color="red", relation="in_room", anchor="northwest room")
        with self.assertRaises(LookupError):
            self.ids(shape="box", relation="nearest_to", anchor="yellow ball")

    def test_survey_policy(self):
        self.assertTrue(needs_survey(FetchStep(color="red", quantifier="all", destination="dock")))
        self.assertTrue(
            needs_survey(FetchStep(color="red", relation="leftmost", destination="dock"))
        )
        self.assertFalse(
            needs_survey(
                FetchStep(color="red", relation="in_room", anchor="west room", destination="dock")
            )
        )


if __name__ == "__main__":
    unittest.main()
