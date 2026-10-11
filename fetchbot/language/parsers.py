"""Turn text like 'bring the red box to the dock, then the green ball to the table'
into a :class:`~fetchbot.language.schema.TaskProgram`.

* :class:`RuleBasedParser` (evaluation baseline, ``--parser rule`` for testing):
  regex/keyword grammar with synonyms, ``then`` sequencing,
  ``all``/``every``, spatial phrases and pronouns ("grab X and take it to Y"). Fast,
  offline, the baseline.
* :class:`LLMParser` (the run-time parser): a local instruction model (llama.cpp,
  GGUF quant, CPU) with JSON-schema-constrained decoding and a few-shot prompt;
  validated, retried once, otherwise the command is rejected.

Synonyms ('crimson crate' -> red box, 'drop-off zone' -> dock) come from the plain
synonym tables below (``COLOR_WORDS``, ``SHAPE_WORDS``, ``PLACE_ALIASES``, ...); the
LLM handles phrasing outside them.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from fetchbot.language.schema import (
    COLORS,
    LANDMARKS,
    PLACES,
    RELATIONS,
    ROOMS,
    SHAPES,
    FetchStep,
    TaskProgram,
    llm_json_schema,
    validate_program,
)
from fetchbot.types import FetchCommand


def program_to_command(program: TaskProgram, index: int = 0) -> FetchCommand:
    """Legacy single-step view of one program step."""
    s = program.steps[index]
    return FetchCommand(
        target_object=s.shape or "object",
        destination=s.destination,
        attributes=(s.color,) if s.color else (),
        raw_text=program.raw_text,
    )


class CommandParser(ABC):
    """Parses free-form text into a structured :class:`TaskProgram`."""

    #: Set after each call: how the last command was parsed (for logs / evaluation).
    last_info: dict

    @abstractmethod
    def parse_program(self, text: str) -> TaskProgram:
        """Return the parsed program, or raise ``ValueError`` if not understood."""

    def parse(self, text: str) -> FetchCommand:
        """Legacy API: the first step as a :class:`FetchCommand`."""
        return program_to_command(self.parse_program(text))


# --------------------------------------------------------------------------- rules
VERBS = (
    "bring",
    "fetch",
    "take",
    "carry",
    "deliver",
    "move",
    "get",
    "put",
    "place",
    "grab",
    "pick",
    "drop",
    "haul",
    "transport",
    "return",
    "relocate",
    "collect",
    "gather",
    "retrieve",
    "shift",
    "drag",
    "push",
    "send",
    "leave",
    "set",
    "bring back",
)
SHAPE_WORDS: dict[str, str] = {
    "box": "box",
    "cube": "box",
    "block": "box",
    "crate": "box",
    "brick": "box",
    "package": "box",
    "carton": "box",
    "chest": "box",
    "ball": "ball",
    "sphere": "ball",
    "orb": "ball",
    "globe": "ball",
    "mug": "cylinder",
    "cup": "cylinder",
    "can": "cylinder",
    "cylinder": "cylinder",
    "bottle": "cylinder",
    "tin": "cylinder",
    "tube": "cylinder",
    "jar": "cylinder",
    "canister": "cylinder",
    "tumbler": "cylinder",
    "thermos": "cylinder",
    "flask": "cylinder",
    "glass": "cylinder",
    "pipe": "cylinder",
    "drum": "cylinder",
    "barrel": "cylinder",
    "parcel": "box",
    "case": "box",
    "container": "box",
    "bin": "box",
    "square": "box",
    "dice": "box",
    "die": "box",
    "toybox": "box",
    "marble": "ball",
    "bead": "ball",
    "bauble": "ball",
    "round": "ball",
    "spherical": "ball",
}
COLOR_WORDS: dict[str, str] = {c: c for c in COLORS} | {
    "scarlet": "red",
    "crimson": "red",
    "ruby": "red",
    "cherry": "red",
    "vermilion": "red",
    "carmine": "red",
    "tomato": "red",
    "reddish": "red",
    "brick-red": "red",
    "navy": "blue",
    "azure": "blue",
    "cobalt": "blue",
    "sapphire": "blue",
    "cerulean": "blue",
    "indigo-blue": "blue",
    "royal": "blue",
    "sky": "blue",
    "bluish": "blue",
    "denim": "blue",
    "emerald": "green",
    "lime": "green",
    "jade": "green",
    "forest": "green",
    "mint": "green",
    "greenish": "green",
    "kelly": "green",
    "sage": "green",
    "golden": "yellow",
    "gold": "yellow",
    "lemon": "yellow",
    "amber": "yellow",
    "goldenrod": "yellow",
    "canary": "yellow",
    "mustard": "yellow",
    "yellowish": "yellow",
    "banana": "yellow",
}
#: Basic color terms the robot does not have: a command using one is rejected
#: (unsatisfiable) instead of being silently mapped to the nearest color.
OTHER_COLORS = frozenset(
    {
        "orange",
        "pink",
        "purple",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "teal",
        "cyan",
        "magenta",
        "violet",
        "indigo",
        "beige",
        "tan",
        "silver",
        "lavender",
        "turquoise",
        "aqua",
        "maroon",
        "olive",
        "khaki",
        "salmon",
        "fuchsia",
    }
)
PLACE_ALIASES: dict[str, str] = {
    "charging dock": "dock",
    "charging station": "dock",
    "charger": "dock",
    "base": "dock",
    "station": "dock",
    "docking station": "dock",
    "home base": "dock",
    "dock": "dock",
    "drop zone": "dock",
    "drop-off zone": "dock",
    "drop off zone": "dock",
    "desk": "table",
    "kitchen table": "table",
    "dining table": "table",
    "table": "table",
    "charging pad": "dock",
    "charging point": "dock",
    "docking area": "dock",
    "drop-off": "dock",
    "drop off": "dock",
    "drop-off point": "dock",
    "drop point": "dock",
    "loading dock": "dock",
    "landing pad": "dock",
    "dinner table": "table",
    "coffee table": "table",
    "work table": "table",
    "workbench": "table",
    "counter": "kitchen",
    "kitchen counter": "kitchen",
    "kitchen area": "kitchen",
    "countertop": "kitchen",
    "worktop": "kitchen",
    "stove": "kitchen",
    "pantry": "kitchen",
    "sink": "kitchen",
    "cooking area": "kitchen",
    "kitchen": "kitchen",
}
LANDMARK_ALIASES: dict[str, str] = {
    "shelf": "shelf",
    "shelves": "shelf",
    "shelving": "shelf",
    "bookshelf": "shelf",
    "rack": "shelf",
    "robot": "robot",
    "you": "robot",
    "yourself": "robot",
    "me": "robot",
}
ROOM_ALIASES: dict[str, str] = {}
for _room in ROOMS:
    _d = _room.split()[0]
    ROOM_ALIASES[_room] = _room
    ROOM_ALIASES[f"{_d} side"] = _room
    ROOM_ALIASES[f"{_d} half"] = _room
    ROOM_ALIASES[f"{_d}ern room"] = _room
    if _d in ("northeast", "northwest", "southeast", "southwest"):
        hy = f"{_d[:5]}-{_d[5:]}"
        ROOM_ALIASES[f"{hy} room"] = _room
ROOM_ALIASES |= {
    "left room": "west room",
    "right room": "east room",
    "left side": "west room",
    "right side": "east room",
    "left half": "west room",
    "right half": "east room",
}
ALL_WORDS = ("all", "every", "each", "both", "everything", "any and all")
EXTREME_WORDS: dict[str, str] = {
    "leftmost": "leftmost",
    "left-most": "leftmost",
    "westmost": "leftmost",
    "westernmost": "leftmost",
    "furthest left": "leftmost",
    "farthest left": "leftmost",
    "most western": "leftmost",
    "on the far left": "leftmost",
    "rightmost": "rightmost",
    "right-most": "rightmost",
    "eastmost": "rightmost",
    "easternmost": "rightmost",
    "furthest right": "rightmost",
    "farthest right": "rightmost",
    "on the far right": "rightmost",
    "northmost": "northmost",
    "northernmost": "northmost",
    "furthest north": "northmost",
    "farthest north": "northmost",
    "topmost": "northmost",
    "southmost": "southmost",
    "southernmost": "southmost",
    "furthest south": "southmost",
    "farthest south": "southmost",
    "bottommost": "southmost",
}
PRONOUNS = ("it", "them", "that one", "those", "these", "both of them", "all of them")
SEPARATORS = (
    r"(?:,?\s*\band then\b|,?\s*\bthen\b|,?\s*\bafter that\b|,?\s*\bafterwards\b|"
    r",\s*next\b|,?\s*\bfinally\b|;|\.\s|,\s*and\b|,)"
)


def _plural_to_singular(word: str) -> str:
    for suffix, rep in (("boxes", "box"), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            cand = word[: -len(suffix)] + rep if suffix != "boxes" else "box"
            if cand in SHAPE_WORDS or cand in ("object", "thing", "item"):
                return cand
    return word


@dataclass
class _Target:
    color: str | None = None
    shape: str | None = None
    quantifier: str = "one"
    relation: str | None = None
    anchor: str | None = None
    plural: bool = False
    words: list[str] = field(default_factory=list)


class RuleBasedParser(CommandParser):
    """Keyword grammar for fetch programs; the cheap, deterministic baseline."""

    VERBS = VERBS
    COLORS = COLORS
    SHAPES = SHAPE_WORDS
    DEFAULT_DESTINATIONS = PLACES
    ALIASES = PLACE_ALIASES

    def __init__(self, destinations: Iterable[str] | None = None) -> None:
        self.destinations = tuple(destinations or self.DEFAULT_DESTINATIONS)
        self.last_info: dict = {}

    @staticmethod
    def known_words() -> set[str]:
        words = set(VERBS) | set(SHAPE_WORDS) | set(COLOR_WORDS) | set(LANDMARK_ALIASES)
        for table in (PLACE_ALIASES, ROOM_ALIASES, EXTREME_WORDS):
            for k in table:
                words.update(k.split())
        words |= {w + "s" for w in SHAPE_WORDS} | {"boxes", "shelves", "everything"}
        return words

    # ---------------------------------------------------------------- top level
    def parse_program(self, text: str) -> TaskProgram:
        t0 = time.perf_counter()
        try:
            prog = self._parse(text, text)
        finally:
            self.last_info = {"parser": "rule", "latency_s": time.perf_counter() - t0}
        return prog

    def _clean(self, text: str) -> str:
        t = text.lower().replace("’", "'")
        t = re.sub(r"[^a-z0-9,;.' -]+", " ", t)
        t = re.sub(
            r"\b(please|could you|can you|would you|kindly|i want you to|i need you to"
            r"|go and|go|now|for me|quickly|also)\b",
            " ",
            t,
        )
        t = re.sub(r"\s+", " ", t).strip(" .")
        return t

    def _parse(self, text: str, raw: str) -> TaskProgram:
        t = self._clean(text)
        if not t:
            raise ValueError("empty command")
        if not any(re.search(rf"\b{v}\b", t) for v in VERBS):
            raise ValueError(f"no fetch verb ({'/'.join(VERBS[:6])}, ...) in {raw!r}")
        clauses = [c.strip(" ,.") for c in re.split(SEPARATORS, t) if c and c.strip(" ,.")]
        # ' and <verb>' also separates clauses ("grab the box and take it to the dock").
        verb_alt = "|".join(VERBS)
        split2: list[str] = []
        for c in clauses:
            split2 += [p.strip() for p in re.split(rf"\band (?=(?:{verb_alt})\b)", c) if p.strip()]
        steps: list[dict] = []
        pending: list[_Target] = []
        for clause in split2:
            dest, obj_part = self._split_destination(clause)
            obj_part = re.sub(rf"^(?:{verb_alt})\b( up| back)?", " ", obj_part).strip()
            obj_part = re.sub(rf"\b(?:{verb_alt})\b( up| back)?$", " ", obj_part).strip()
            obj_part = re.sub(
                r"\b(up|back|over|along|here|there|with you)\b$", "", obj_part
            ).strip()
            pron = obj_part in PRONOUNS or obj_part == ""
            if dest is None:
                if pron:
                    raise ValueError(f"nothing to fetch in {clause!r}")
                pending += self._targets(obj_part, raw)
                continue
            if pron:
                if not pending:
                    raise ValueError(f"'{obj_part or 'it'}' refers to nothing in {raw!r}")
                targets, pending = pending, []
            else:
                if pending:
                    raise ValueError(f"no destination for {[p.words for p in pending]}")
                targets = self._targets(obj_part, raw)
            for tg in targets:
                steps.append(
                    {
                        "color": tg.color,
                        "shape": tg.shape,
                        "quantifier": tg.quantifier,
                        "relation": tg.relation,
                        "anchor": tg.anchor,
                        "destination": dest,
                    }
                )
        if pending:
            raise ValueError(f"no destination ('... to the <place>') in {raw!r}")
        if not steps:
            raise ValueError(f"no fetch step found in {raw!r}")
        return validate_program({"steps": steps}, raw_text=raw, places=self.destinations)

    # ---------------------------------------------------------------- pieces
    def _place_alias_items(self):
        aliases = {k: v for k, v in PLACE_ALIASES.items() if v in self.destinations}
        aliases |= {d: d for d in self.destinations}
        return sorted(aliases.items(), key=lambda kv: -len(kv[0]))

    def _split_destination(self, clause: str) -> tuple[str | None, str]:
        """Find the last '<prep> [the] <place>' not governed by a relation word."""
        best = None
        for alias, name in self._place_alias_items():
            for m in re.finditer(
                rf"\b(to|into|onto|in|at|on|by|over to|up to|back to|"
                rf"toward|towards)\s+(?:the\s+|my\s+)?{re.escape(alias)}\b",
                clause,
            ):
                before = clause[: m.start()].split()
                if before and before[-1] in (
                    "nearest",
                    "closest",
                    "near",
                    "next",
                    "farthest",
                    "furthest",
                    "from",
                    "away",
                    "beside",
                ):
                    continue
                if best is None or m.start() > best[0]:
                    best = (m.start(), m.end(), name)
        if best is None:
            # A trailing unknown place ("... to the garage") is an error, not an object.
            m = re.search(r"\b(?:to|into|onto)\s+(?:the\s+)?([a-z' -]+)$", clause)
            if m and not any(
                w in m.group(1).split()
                for w in list(SHAPE_WORDS) + ["object", "thing", "it", "them"]
            ):
                words = m.group(1).split()
                before = clause[: m.start()].split()
                if not (
                    before
                    and before[-1] in ("nearest", "closest", "next", "farthest", "furthest", "near")
                ):
                    raise ValueError(
                        f"unknown destination {m.group(1)!r}; known: {self.destinations}"
                        if words
                        else "no destination"
                    )
            return None, clause
        s, e, name = best
        return name, (clause[:s] + " " + clause[e:]).strip()

    def _targets(self, obj_part: str, raw: str) -> list[_Target]:
        # 'the red box and the green ball' -> two targets sharing the destination.
        parts = [p.strip() for p in re.split(r"\b(?:and|as well as|plus)\b", obj_part) if p.strip()]
        out = [self._target(p, raw) for p in parts]
        return out

    def _anchor(self, phrase: str) -> str | None:
        words = phrase.replace("the ", " ").split()
        p = " ".join(words)
        for alias, name in self._place_alias_items():
            if re.search(rf"\b{re.escape(alias)}\b", p):
                return name
        for alias, name in LANDMARK_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", p):
                return name
        color = next((COLOR_WORDS[w] for w in words if w in COLOR_WORDS), None)
        shape = next(
            (
                SHAPE_WORDS[_plural_to_singular(w)]
                for w in words
                if _plural_to_singular(w) in SHAPE_WORDS
            ),
            None,
        )
        if color and shape:
            return f"{color} {shape}"
        return color or shape

    def _target(self, part: str, raw: str) -> _Target:
        tg = _Target()
        p = " " + part + " "
        # Rooms: 'in/from the west room'.
        for alias, room in sorted(ROOM_ALIASES.items(), key=lambda kv: -len(kv[0])):
            m = re.search(rf"\b(?:in|from|inside|within|on)\s+(?:the\s+)?{re.escape(alias)}\b", p)
            if m:
                tg.relation, tg.anchor = "in_room", room
                p = p[: m.start()] + " " + p[m.end() :]
                break
        if tg.relation is None:
            for alias, rel in sorted(EXTREME_WORDS.items(), key=lambda kv: -len(kv[0])):
                m = re.search(rf"\b{re.escape(alias)}\b", p)
                if m:
                    tg.relation = rel
                    p = p[: m.start()] + " " + p[m.end() :]
                    break
        if tg.relation is None:
            adj = re.search(
                r"\b(nearest|closest|farthest|furthest)\s+(?!to\b|from\b|the\b|"
                r"away\b|of\b|my\b)",
                p,
            )
            m = (
                None
                if adj
                else re.search(
                    r"\b(nearest|closest|next|near|beside|farthest|furthest)\b"
                    r"(?:\s+(?:to|from|away from|of))?\s+(.*)$",
                    p,
                )
            )
            if adj:  # 'the nearest box' (adjective, no anchor) -> nearest to the robot
                tg.relation = (
                    "farthest_from" if adj.group(1) in ("farthest", "furthest") else "nearest_to"
                )
                tg.anchor = "robot"
                p = p[: adj.start()] + " " + p[adj.end() :]
            if m:
                rel = "farthest_from" if m.group(1) in ("farthest", "furthest") else "nearest_to"
                anchor = self._anchor(m.group(2))
                if anchor is None:
                    # 'the nearest box' (no anchor) -> nearest to the robot.
                    anchor_words = m.group(2).split()
                    if (
                        any(
                            _plural_to_singular(w) in SHAPE_WORDS
                            or w in COLOR_WORDS
                            or w in ("object", "thing", "one", "item")
                            for w in anchor_words
                        )
                        or not anchor_words
                    ):
                        tg.relation, tg.anchor = rel, "robot"
                        p = p[: m.start()] + " " + m.group(2)
                    else:
                        raise ValueError(f"unknown landmark {m.group(2).strip()!r}")
                else:
                    tg.relation, tg.anchor = rel, anchor
                    p = p[: m.start()]
        words = p.split()
        tg.words = words
        bad = [w for w in words if w in OTHER_COLORS]
        if bad:
            raise ValueError(f"unsupported color {bad[0]!r}; known: {COLORS}")
        if any(w in ALL_WORDS for w in words) or re.search(r"\bany and all\b", p):
            tg.quantifier = "all"
        for w in words:
            if w in COLOR_WORDS and tg.color is None:
                tg.color = COLOR_WORDS[w]
            s = _plural_to_singular(w)
            if s in SHAPE_WORDS and tg.shape is None:
                tg.shape = SHAPE_WORDS[s]
                tg.plural = tg.plural or s != w
            elif s in ("object", "thing", "item", "toy") and s != w:
                tg.plural = True
        if tg.plural and tg.quantifier == "one" and "the" in words and tg.relation is None:
            tg.quantifier = "all"  # 'bring the red boxes to the dock'
        if tg.color is None and tg.shape is None and tg.quantifier == "one":
            raise ValueError(f"no object color or shape in {raw!r}")
        if tg.relation == "nearest_to" and tg.anchor == "robot" and tg.quantifier == "all":
            tg.relation, tg.anchor = None, None
        return tg

    def _match_destination(self, dest: str) -> str | None:  # kept for API compatibility
        for alias, name in self._place_alias_items():
            if re.search(rf"\b{re.escape(alias)}\b", dest):
                return name
        return None


# --------------------------------------------------------------------------- LLM
SYSTEM_PROMPT = """You convert a robot command into a JSON task program. The robot fetches \
objects and drops them at named places. Output only JSON.

Program: {"steps": [step, ...]} in the order the user wants them done. A step has these \
keys; copy the color and shape the command names, and leave a key out unless the \
command clearly says it (never guess):
- "color": red, green, blue or yellow (map shades: crimson/scarlet -> red, navy/azure -> \
blue, lime/emerald -> green, gold/lemon -> yellow)
- "shape": box (box/cube/crate/block/carton), cylinder (mug/cup/can/bottle/tin/jar), \
ball (ball/sphere/globe/orb); leave out for "object"/"thing"/"item"
- "quantifier": "all" for all/every/each/both or plurals ("the red boxes")
- "relation" (only if the command has one): nearest_to or farthest_from (with "anchor"), \
leftmost, rightmost, northmost, southmost (no anchor), in_room (anchor = a room)
- "anchor": a place (dock, table, kitchen), "shelf", "robot" ("the nearest box" with no \
landmark), an object such as "blue cylinder", or a room (west room, east room, \
northwest room, southwest room, northeast room, southeast room)
- "destination" (always): dock (dock/charger/base/station), table (table/desk), \
kitchen (kitchen/counter/pantry)
"it"/"them" refers to the objects just mentioned. Objects joined by "and" with one \
destination become separate steps.
If the command is not a fetch task or uses an unknown color, shape or place (e.g. purple, \
the moon), output {"steps": []}."""

#: Replaces the last paragraph of SYSTEM_PROMPT when ``LLMConfig.near_synonyms`` is on.
#: (Written after looking at the misses on the first held-out set, so that set is not
#: blind for it; ``commands_heldout2.json`` was frozen before.)
NEAR_SYNONYM_RULE = """Map any other shade, object word or place word to the closest \
allowed value (e.g. maroon -> red, canister -> cylinder, cooking area -> kitchen). Output \
{"steps": []} only if it is not a fetch task, has no destination, or names a color, shape \
or place that is clearly none of the allowed ones (e.g. purple, black, the moon)."""

EMPTY_PLAN_FEEDBACK = (
    "Your last answer was an empty plan. If the command asks to fetch something with the "
    "known colors, shapes and places, return that plan; if it truly does not, return "
    '{"steps": []} again.'
)

FEW_SHOT: list[tuple[str, dict]] = [
    (
        "bring the red box to the dock",
        {
            "steps": [
                {
                    "color": "red",
                    "shape": "box",
                    "quantifier": "one",
                    "relation": None,
                    "anchor": None,
                    "destination": "dock",
                }
            ]
        },
    ),
    (
        "grab the green mug and take it to the kitchen, then bring the yellow sphere to the desk",
        {
            "steps": [
                {
                    "color": "green",
                    "shape": "cylinder",
                    "quantifier": "one",
                    "relation": None,
                    "anchor": None,
                    "destination": "kitchen",
                },
                {
                    "color": "yellow",
                    "shape": "ball",
                    "quantifier": "one",
                    "relation": None,
                    "anchor": None,
                    "destination": "table",
                },
            ]
        },
    ),
    (
        "move every blue object to the charging dock",
        {
            "steps": [
                {
                    "color": "blue",
                    "shape": None,
                    "quantifier": "all",
                    "relation": None,
                    "anchor": None,
                    "destination": "dock",
                }
            ]
        },
    ),
    (
        "fetch the ball closest to the shelf and put it on the table",
        {
            "steps": [
                {
                    "color": None,
                    "shape": "ball",
                    "quantifier": "one",
                    "relation": "nearest_to",
                    "anchor": "shelf",
                    "destination": "table",
                }
            ]
        },
    ),
    (
        "take the leftmost cube to the kitchen",
        {
            "steps": [
                {
                    "color": None,
                    "shape": "box",
                    "quantifier": "one",
                    "relation": "leftmost",
                    "anchor": None,
                    "destination": "kitchen",
                }
            ]
        },
    ),
    (
        "carry all the yellow things in the east room to the dock",
        {
            "steps": [
                {
                    "color": "yellow",
                    "shape": None,
                    "quantifier": "all",
                    "relation": "in_room",
                    "anchor": "east room",
                    "destination": "dock",
                }
            ]
        },
    ),
    (
        "bring the cube nearest the dock to the kitchen",
        {
            "steps": [
                {
                    "color": None,
                    "shape": "box",
                    "quantifier": "one",
                    "relation": "nearest_to",
                    "anchor": "dock",
                    "destination": "kitchen",
                }
            ]
        },
    ),
    ("bring the purple ball to the dock", {"steps": []}),
]


@dataclass
class LLMConfig:
    """Where the GGUF model is and how to run it (all overridable by env vars).

    Default: Gemma 4 E2B-it, unquantized BF16 GGUF (``gemma-4-E2B-it-BF16.gguf``,
    ~9.3 GB file, memory-mapped; ~0.4 s per command on an RTX 4090 through
    ``llama-server``, ~21 s on 4 CPU threads; comparison with Qwen2.5-1.5B in the
    README). Any other instruct GGUF is a drop-in swap via ``FETCHBOT_LLM_MODEL``.
    """

    model_path: str = os.environ.get(
        "FETCHBOT_LLM_MODEL",
        str(Path(os.environ.get("FETCHBOT_MODEL_DIR", "models")) / "gemma-4-E2B-it-BF16.gguf"),
    )
    n_ctx: int = 4096
    n_threads: int = int(os.environ.get("FETCHBOT_LLM_THREADS", "4"))
    n_gpu_layers: int = int(os.environ.get("FETCHBOT_LLM_GPU_LAYERS", "0"))
    max_tokens: int = 400
    #: Sampling recommended by the Gemma 4 model card for all use cases (no seed
    #: pinning, so outputs vary run to run; the retry with error feedback catches
    #: invalid samples).
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    grammar: bool = True  # JSON-schema-constrained decoding
    seed: int | None = None
    #: Gemma 4 thinking mode (llama-server only): the model reasons freely first
    #: (``chat_template_kwargs.enable_thinking``), then the final answer is extracted as
    #: JSON and validated; no grammar constraint in that mode.
    thinking: bool = os.environ.get("FETCHBOT_LLM_THINKING", "0") == "1"
    thinking_max_tokens: int = 2048
    #: Prompt asks to map near-synonyms to the closest known value, and the grounding
    #: check accepts a color/shape the command only describes with an unknown word.
    near_synonyms: bool = os.environ.get("FETCHBOT_LLM_NEAR_SYNONYMS", "0") == "1"
    #: Optional llama.cpp ``llama-server`` (OpenAI-compatible) base URL, e.g.
    #: ``http://127.0.0.1:8080``: same prompt and JSON schema, model run by the server
    #: (e.g. on a GPU). Empty = load ``model_path`` in-process with llama-cpp-python.
    server_url: str = os.environ.get("FETCHBOT_LLM_SERVER", "")


def _compact(prog: dict) -> dict:
    """Drop null / default keys: the model only writes what the command says."""
    keep = []
    for st in prog["steps"]:
        keep.append(
            {
                k: v
                for k, v in st.items()
                if v is not None and not (k == "quantifier" and v == "one")
            }
        )
    return {"steps": keep}


class LLMParser(CommandParser):
    """Local LLM parser (llama.cpp) with schema-constrained JSON output.

    ``parse_program``: LLM -> strict schema validation + grounding check (every color
    and shape in the program must be named in the command). If that fails, the model
    is asked once more with the error message; if the second answer also fails, the
    command is rejected with ``ValueError``. There is no rule-based fallback at run
    time (the rule parser is only the evaluation baseline). An empty program
    (``{"steps": []}``) is the model's way to say it cannot do the command.
    ``last_info`` records the raw outputs, attempts and latency.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        destinations: Iterable[str] | None = None,
        retries: int = 2,
    ) -> None:
        self.config = config or LLMConfig()
        self.destinations = tuple(destinations or PLACES)
        self.retries = retries
        self._llm = None
        self.load_error = ""
        self.last_info: dict = {}

    # ------------------------------------------------------------------ model
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._llm is not None or self.load_error:
            return self._llm
        if self.config.server_url:
            self._llm = "server"
            return self._llm
        if not Path(self.config.model_path).is_file():
            self.load_error = (
                f"model file not found: {self.config.model_path} (download it, "
                "see README 'Language model', or set FETCHBOT_LLM_MODEL)"
            )
            return None
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            self.load_error = (
                f"llama-cpp-python is not installed ({exc}); pip install llama-cpp-python"
            )
            return None
        self._llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
            **({} if self.config.seed is None else {"seed": self.config.seed}),
        )
        return self._llm

    def _messages(
        self, text: str, error: str | None = None, previous: str | None = None
    ) -> list[dict]:
        system = SYSTEM_PROMPT
        if self.config.near_synonyms:
            system = system[: system.index("If the command is not a fetch task")] + (
                NEAR_SYNONYM_RULE
            )
        msgs = [{"role": "system", "content": system}]
        for cmd, prog in FEW_SHOT:
            msgs.append({"role": "user", "content": cmd})
            msgs.append({"role": "assistant", "content": json.dumps(_compact(prog))})
        msgs.append({"role": "user", "content": text})
        if error:
            # Keep user/assistant turns alternating (Gemma's chat template requires it).
            msgs.append({"role": "assistant", "content": previous or "{}"})
            msgs.append(
                {
                    "role": "user",
                    "content": error
                    if error == EMPTY_PLAN_FEEDBACK
                    else f"Your last answer was invalid: {error}. Answer again with valid JSON.",
                }
            )
        return msgs

    def _query(self, text: str, error: str | None = None, previous: str | None = None) -> str:
        llm = self._load()
        if llm is None:
            raise RuntimeError(self.load_error)
        if self.config.server_url:
            return self._query_server(text, error, previous)
        if self.config.thinking:
            raise RuntimeError("thinking mode needs a llama-server (FETCHBOT_LLM_SERVER)")
        kw = {}
        if self.config.grammar:
            kw["response_format"] = {"type": "json_object", "schema": llm_json_schema()}
        out = llm.create_chat_completion(
            messages=self._messages(text, error, previous),
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            **kw,
        )
        return out["choices"][0]["message"]["content"]

    def _query_server(
        self, text: str, error: str | None = None, previous: str | None = None
    ) -> str:
        import urllib.request

        body = {
            "messages": self._messages(text, error, previous),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "chat_template_kwargs": {"enable_thinking": self.config.thinking},
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        if self.config.thinking:
            body["max_tokens"] = self.config.thinking_max_tokens
        elif self.config.grammar:
            body["response_format"] = {"type": "json_object", "schema": llm_json_schema()}
        req = urllib.request.Request(
            self.config.server_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read())
        except OSError as exc:
            raise RuntimeError(f"llama-server at {self.config.server_url} failed: {exc}") from exc
        msg = out["choices"][0]["message"]
        self._reasoning = msg.get("reasoning_content") or ""
        return msg.get("content") or ""

    # ------------------------------------------------------------------ parse
    def parse_program(self, text: str) -> TaskProgram:
        """Parse ``text``; raise ``ValueError`` if rejected, ``RuntimeError`` if no model."""
        t0 = time.perf_counter()
        info = {"parser": "llm", "attempts": 0, "raw_outputs": [], "reasoning": []}
        error = None
        out = None
        try:
            for _attempt in range(1 + self.retries):
                info["attempts"] += 1
                self._reasoning = ""
                out = self._query(text, error, out if error else None)
                info["raw_outputs"].append(out)
                info["reasoning"].append(self._reasoning)
                if not out.strip():
                    info["blank"] = info.get("blank", 0) + 1
                try:
                    data = json.loads(_json_part(out))
                except json.JSONDecodeError as exc:
                    error = f"not JSON ({exc})"
                    continue
                if isinstance(data, dict) and data.get("steps") == []:
                    # Sampled outputs vary, so an empty plan is retried like an error.
                    info["empty"] = info.get("empty", 0) + 1
                    error = EMPTY_PLAN_FEEDBACK
                    continue
                try:
                    prog = validate_program(data, raw_text=text, places=self.destinations)
                    ungrounded = ungrounded_words(prog, text, lenient=self.config.near_synonyms)
                    if ungrounded:
                        raise ValueError(f"{ungrounded} not mentioned in the command")
                except ValueError as exc:
                    error = str(exc)[:300]
                    info["error"] = error
                    continue
                return prog
            if error == EMPTY_PLAN_FEEDBACK:
                info["rejected"] = "model"
                raise ValueError(
                    f"the language model rejected {text!r} (no fetch task with the known "
                    f"colors, shapes and places, {info['attempts']} attempts)"
                )
            info["rejected"] = "invalid"
            raise ValueError(
                f"could not parse {text!r}: the model's output stayed invalid "
                f"after {info['attempts']} attempts ({error})"
            )
        finally:
            info["latency_s"] = time.perf_counter() - t0
            self.last_info = info


def _json_part(out: str) -> str:
    """The JSON object in a model answer (strips code fences / text around it)."""
    a, b = out.find("{"), out.rfind("}")
    return out[a : b + 1] if 0 <= a < b else out


#: Words that are neither vocabulary nor descriptions (for the lenient grounding check).
_COMMON = frozenset(
    "a an the to of and then it them me please robot can could you would go get take bring "
    "put move carry fetch grab deliver drop leave set haul place pick up off over into onto "
    "in on at by from with for next after that afterwards first second finally also too "
    "all every each both any thing things object objects item items stuff one is are be "
    "should needs need goes end colored coloured ish thanks there this those these which "
    "who its back down out way i want".split()
)


def _known_words() -> frozenset[str]:
    words = set(_COMMON) | set(COLOR_WORDS) | set(SHAPE_WORDS) | set(OTHER_COLORS)
    for table in (PLACE_ALIASES, LANDMARK_ALIASES, ROOM_ALIASES, EXTREME_WORDS):
        for k in table:
            words.update(re.findall(r"[a-z]+", k))
    for v in VERBS + ALL_WORDS + PRONOUNS:
        words.update(re.findall(r"[a-z]+", v))
    for r in ROOMS + RELATIONS:
        words.update(re.findall(r"[a-z]+", r))
    words.update(("nearest", "closest", "farthest", "furthest", "near", "far", "away"))
    return frozenset(words)


def ungrounded_words(program: TaskProgram, text: str, lenient: bool = False) -> list[str]:
    """Colors/shapes in ``program`` that no word of ``text`` maps to (hallucinations).

    Words outside the synonym tables (e.g. 'vermilion') are not grounded, so the LLM
    parser rejects such commands even if the model mapped them correctly.
    """
    words = re.findall(r"[a-z]+", text.lower())
    colors = {COLOR_WORDS.get(w) for w in words}
    shapes = {SHAPE_WORDS.get(_plural_to_singular(w)) for w in words}
    # Round/spherical things are balls; this is a description, not a hallucination.
    if {"round", "spherical"} & set(words):
        shapes.add("ball")
    if lenient and not (set(words) & OTHER_COLORS):
        # A color/shape that the command only describes with a word outside the tables
        # (e.g. 'carmine chest') is accepted, if there is such a word at all.
        known = _known_words()
        if any(_plural_to_singular(w) not in known and w not in known for w in words):
            colors.update(COLORS)
            shapes.update(SHAPES)
    bad = []
    for s in program.steps:
        if s.color and s.color not in colors:
            bad.append(s.color)
        if s.shape and s.shape not in shapes:
            bad.append(s.shape)
    return bad


def make_parser(
    kind: str = "llm", destinations: Iterable[str] | None = None, config: LLMConfig | None = None
) -> CommandParser:
    """Factory used by the CLI: ``kind`` is ``rule`` or ``llm``."""
    if kind == "rule":
        return RuleBasedParser(destinations=destinations)
    if kind == "llm":
        return LLMParser(config=config, destinations=destinations)
    raise ValueError(f"unknown parser {kind!r} (rule | llm)")


__all__ = [
    "CommandParser",
    "FetchStep",
    "LANDMARKS",
    "LLMConfig",
    "LLMParser",
    "RuleBasedParser",
    "TaskProgram",
    "make_parser",
    "program_to_command",
]
