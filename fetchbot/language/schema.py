"""Task-program schema: what a parsed command means, independent of the parser.

A command becomes a :class:`TaskProgram`, an ordered list of :class:`FetchStep`.
Each step is ``fetch(target selector, destination)``::

    {"steps": [{"color": "red", "shape": "box", "quantifier": "one",
                "relation": "nearest_to", "anchor": "shelf", "destination": "table"}]}

* ``color`` / ``shape``: optional filters (at least one, unless ``quantifier`` is
  ``all``: "bring everything to the dock").
* ``quantifier``: ``one`` (a single object) or ``all`` (every matching object).
* ``relation`` + ``anchor``: optional spatial selector, evaluated in the map frame
  (x = east = right on the map panel, y = north = up):

  - ``nearest_to`` / ``farthest_from`` + anchor: a place (``dock``, ``table``,
    ``kitchen``), ``shelf`` (the nearest shelf), ``robot`` (where the robot is when
    the step starts) or an object ``"<color> <shape>"`` such as ``"blue cylinder"``;
  - ``leftmost`` / ``rightmost`` / ``northmost`` / ``southmost`` (no anchor);
  - ``in_room`` + a room name (``west room``, ``northeast room``, ...).

* ``destination``: a named drop zone.

The schema is strict (pydantic, ``extra="forbid"``): anything else is rejected
with ``ValueError`` so a parser can retry or reject the command.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)

COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow")
SHAPES: tuple[str, ...] = ("box", "cylinder", "ball")
PLACES: tuple[str, ...] = ("dock", "table", "kitchen")
#: Rooms of the randomized arena (see SceneRandomizer): a vertical wall makes a west
#: and an east half, and one half is split again into a north and a south room.
ROOMS: tuple[str, ...] = (
    "west room",
    "east room",
    "northwest room",
    "southwest room",
    "northeast room",
    "southeast room",
)
DISTANCE_RELATIONS: tuple[str, ...] = ("nearest_to", "farthest_from")
EXTREME_RELATIONS: tuple[str, ...] = ("leftmost", "rightmost", "northmost", "southmost")
RELATIONS: tuple[str, ...] = DISTANCE_RELATIONS + EXTREME_RELATIONS + ("in_room",)
LANDMARKS: tuple[str, ...] = PLACES + ("shelf", "robot")
OBJECT_ANCHORS: tuple[str, ...] = (
    tuple(f"{c} {s}" for c in COLORS for s in SHAPES) + COLORS + SHAPES
)
ANCHORS: tuple[str, ...] = LANDMARKS + OBJECT_ANCHORS + ROOMS
MAX_STEPS = 6

Color = Literal["red", "green", "blue", "yellow"]
Shape = Literal["box", "cylinder", "ball"]
Relation = Literal[
    "nearest_to", "farthest_from", "leftmost", "rightmost", "northmost", "southmost", "in_room"
]


class FetchStep(BaseModel):
    """One ``fetch(target, destination)`` step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    color: Color | None = None
    shape: Shape | None = None
    quantifier: Literal["one", "all"] = "one"
    relation: Relation | None = None
    anchor: str | None = None
    destination: str

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> FetchStep:
        places = tuple((info.context or {}).get("places") or PLACES)
        if self.destination not in places:
            raise ValueError(f"unknown destination {self.destination!r}; known: {places}")
        if self.color is None and self.shape is None and self.quantifier == "one":
            raise ValueError("a 'one' target needs a color or a shape")
        rel, anc = self.relation, self.anchor
        if rel is None:
            if anc is not None:
                raise ValueError("anchor given without a relation")
        elif rel in DISTANCE_RELATIONS:
            if anc not in LANDMARKS + OBJECT_ANCHORS:
                raise ValueError(f"{rel} needs a landmark or object anchor, got {anc!r}")
            if anc in OBJECT_ANCHORS and _anchor_matches_target(anc, self):
                raise ValueError(f"anchor {anc!r} is the target itself")
        elif rel in EXTREME_RELATIONS:
            if anc is not None:
                raise ValueError(f"{rel} takes no anchor")
        elif rel == "in_room" and anc not in ROOMS:
            raise ValueError(f"in_room needs a room in {ROOMS}, got {anc!r}")
        if self.quantifier == "all" and rel is not None and rel != "in_room":
            raise ValueError(f"'all' cannot be combined with the superlative {rel}")
        return self

    @property
    def anchor_object(self) -> tuple[str | None, str | None] | None:
        """``(color, shape)`` if the anchor is an object, else ``None``."""
        if self.anchor not in OBJECT_ANCHORS:
            return None
        words = self.anchor.split()
        color = next((w for w in words if w in COLORS), None)
        shape = next((w for w in words if w in SHAPES), None)
        return color, shape

    def describe(self) -> str:
        """Short human-readable form, e.g. ``all red -> dock``."""
        what = " ".join(w for w in (self.color, self.shape) if w) or "object"
        if self.quantifier == "all":
            what = (
                f"all {what}{'es' if self.shape == 'box' else 's'}"
                if self.shape
                else f"all {what} objects"
                if self.color
                else "all objects"
            )
        if self.relation:
            what += f" [{self.relation}{' ' + self.anchor if self.anchor else ''}]"
        return f"fetch({what} -> {self.destination})"


def _anchor_matches_target(anchor: str, step: FetchStep) -> bool:
    words = anchor.split()
    a_col = next((w for w in words if w in COLORS), None)
    a_shp = next((w for w in words if w in SHAPES), None)
    return a_col == step.color and a_shp == step.shape and a_col is not None and a_shp is not None


class TaskProgram(BaseModel):
    """An ordered list of fetch steps parsed from one command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: list[FetchStep] = Field(min_length=1, max_length=MAX_STEPS)
    raw_text: str = Field(default="", exclude=True)

    def describe(self) -> str:
        return "; ".join(f"{i + 1}. {s.describe()}" for i, s in enumerate(self.steps))

    def canonical(self) -> list[dict]:
        """Comparable form (used for exact-match accuracy)."""
        return [s.model_dump() for s in self.steps]


def validate_program(
    data: dict | str, raw_text: str = "", places: tuple[str, ...] | None = None
) -> TaskProgram:
    """Validate parser output (dict or JSON text); raise ``ValueError`` if invalid.

    ``places`` overrides the allowed destinations (default :data:`PLACES`)."""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("program must be a JSON object")
        prog = TaskProgram.model_validate(data, context={"places": places})
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid task program: {exc}") from None
    return prog.model_copy(update={"raw_text": raw_text})


def llm_json_schema() -> dict:
    """Flat JSON schema for grammar-constrained decoding (vocabulary as enums).

    Cross-field rules (e.g. anchors per relation) are checked afterwards by
    :func:`validate_program`. ``minItems`` is 0 so the model can reject a command
    with ``{"steps": []}``.
    """

    def nullable_enum(values):
        return {"enum": list(values)}

    step = {
        "type": "object",
        "properties": {
            "color": nullable_enum(COLORS),
            "shape": nullable_enum(SHAPES),
            "quantifier": {"enum": ["one", "all"]},
            "relation": nullable_enum(RELATIONS),
            "anchor": nullable_enum(ANCHORS),
            "destination": {"enum": list(PLACES)},
        },
        # All keys optional (omitted = not said), in sentence order; a missing
        # destination is caught by the schema validation afterwards.
        "required": [],
    }
    return {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": step, "minItems": 0, "maxItems": MAX_STEPS}
        },
        "required": ["steps"],
    }
