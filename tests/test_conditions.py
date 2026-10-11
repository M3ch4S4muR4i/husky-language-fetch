"""Scene generation under the harder evaluation conditions (no physics needed)."""

import math

import pytest

from fetchbot.world import CONDITIONS, SceneRandomizer
from fetchbot.world.difficulty import get_condition
from fetchbot.world.randomizer import CATALOGS


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_every_condition_generates_solvable_scenes(name):
    for seed in range(3):
        spec = SceneRandomizer(seed=seed, difficulty=CONDITIONS[name]).randomize()
        n = len(CATALOGS[CONDITIONS[name].object_set])
        assert len(spec.objects) == n and len(spec.zones) == 3
        assert {r.name for r in spec.regions} >= {"west room", "east room"}
        assert len(spec.regions) == 4


def test_shapes_condition_has_same_color_different_shapes():
    spec = SceneRandomizer(seed=0, difficulty=get_condition("shapes")).randomize()
    reds = {o.shape for o in spec.objects if o.color == "red"}
    assert reds == {"box", "ball"}


def test_regions_cover_the_arena():
    spec = SceneRandomizer(seed=1).randomize()
    w, h = spec.size
    west, east = spec.region("west room"), spec.region("east room")
    assert west.xmax == east.xmin and west.xmin == 0.0 and east.xmax == w
    quads = [r for r in spec.regions if r.name not in ("west room", "east room")]
    assert sum((r.xmax - r.xmin) * (r.ymax - r.ymin) for r in quads) == pytest.approx(
        (west.xmax * h) if quads[0].xmin == 0.0 else (w - east.xmin) * h
    )


def test_narrow_doors_are_narrower():
    def door_gaps(spec):
        segs = sorted(
            (b for b in spec.boxes if b.name.startswith("div_v")), key=lambda b: b.center[1]
        )
        return [
            round((b.center[1] - b.size[1] / 2) - (a.center[1] + a.size[1] / 2), 2)
            for a, b in zip(segs, segs[1:], strict=False)
        ]

    std = SceneRandomizer(seed=1).randomize()
    narrow = SceneRandomizer(seed=1, difficulty=CONDITIONS["narrow_doors"]).randomize()
    assert all(math.isclose(g, 1.6, abs_tol=0.01) for g in door_gaps(std))
    assert all(math.isclose(g, 1.10, abs_tol=0.01) for g in door_gaps(narrow))


def test_distractors_and_person_in_hard_set():
    spec = SceneRandomizer(seed=4, difficulty=CONDITIONS["hard_all"]).randomize()
    names = [b.name for b in spec.boxes] + [c.name for c in spec.cylinders]
    assert sum(n.startswith("distractor") for n in names) == 5
    assert len(spec.actors) == 1


def test_unknown_condition():
    with pytest.raises(ValueError):
        get_condition("impossible")
