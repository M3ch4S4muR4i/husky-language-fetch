"""Command language: schema validation, RuleBasedParser, LLMParser (model tests skippable)."""

import json
import os
import unittest

from fetchbot.language import (
    LLMConfig,
    LLMParser,
    RuleBasedParser,
    TaskProgram,
    llm_json_schema,
    validate_program,
)
from fetchbot.language.schema import FetchStep

MODEL = LLMConfig().model_path


def steps(prog: TaskProgram) -> list[dict]:
    return prog.canonical()


class TestRuleBasedParser(unittest.TestCase):
    def setUp(self):
        self.parser = RuleBasedParser()

    def test_basic_command(self):
        cmd = self.parser.parse("bring the red box to the dock")
        self.assertEqual(cmd.target_object, "box")
        self.assertEqual(cmd.attributes, ("red",))
        self.assertEqual(cmd.destination, "dock")
        self.assertEqual(cmd.raw_text, "bring the red box to the dock")

    def test_synonyms_and_punctuation(self):
        cmd = self.parser.parse("Please fetch me the BLUE mug to the kitchen!")
        self.assertEqual(
            (cmd.target_object, cmd.attributes, cmd.destination), ("cylinder", ("blue",), "kitchen")
        )
        cmd = self.parser.parse("take the green sphere to the desk")
        self.assertEqual((cmd.target_object, cmd.destination), ("ball", "table"))

    def test_color_only(self):
        cmd = self.parser.parse("bring the yellow thing to the table")
        self.assertEqual((cmd.target_object, cmd.attributes), ("object", ("yellow",)))

    def test_rejects_bad_commands(self):
        for text in [
            "hello robot",
            "bring the red box",
            "bring the red box to the moon",
            "bring it to the dock",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parser.parse(text)

    def test_custom_destinations(self):
        parser = RuleBasedParser(destinations=["garage"])
        self.assertEqual(parser.parse("move the red box into the garage").destination, "garage")

    def test_multi_step_and_pronouns(self):
        prog = self.parser.parse_program(
            "grab the red box and take it to the dock, then bring the green ball to the table"
        )
        self.assertEqual(
            prog.describe(), "1. fetch(red box -> dock); 2. fetch(green ball -> table)"
        )

    def test_all_quantifier(self):
        prog = self.parser.parse_program("bring every red object to the dock")
        self.assertEqual(
            steps(prog),
            [
                {
                    "color": "red",
                    "shape": None,
                    "quantifier": "all",
                    "relation": None,
                    "anchor": None,
                    "destination": "dock",
                }
            ],
        )
        prog = self.parser.parse_program("put all the boxes in the kitchen")
        self.assertEqual((prog.steps[0].shape, prog.steps[0].quantifier), ("box", "all"))

    def test_spatial_relations(self):
        s = self.parser.parse_program("bring the box nearest the shelf to the table").steps[0]
        self.assertEqual((s.shape, s.relation, s.anchor), ("box", "nearest_to", "shelf"))
        s = self.parser.parse_program("bring the nearest box to the table").steps[0]
        self.assertEqual((s.relation, s.anchor), ("nearest_to", "robot"))
        s = self.parser.parse_program("fetch the leftmost ball to the dock").steps[0]
        self.assertEqual(s.relation, "leftmost")
        s = self.parser.parse_program("take the red box in the west room to the dock").steps[0]
        self.assertEqual((s.relation, s.anchor), ("in_room", "west room"))

    def test_two_objects_one_destination(self):
        prog = self.parser.parse_program("bring the red box and the green ball to the dock")
        self.assertEqual(len(prog.steps), 2)
        self.assertEqual({s.destination for s in prog.steps}, {"dock"})


class TestSchema(unittest.TestCase):
    def test_valid_program(self):
        prog = validate_program(
            {"steps": [{"color": "red", "shape": "box", "destination": "dock"}]}
        )
        self.assertEqual(prog.steps[0].quantifier, "one")

    def test_rejects_invalid(self):
        bad = [
            {"steps": []},  # empty
            {"steps": [{"color": "purple", "destination": "dock"}]},  # unknown color
            {"steps": [{"color": "red", "destination": "moon"}]},  # unknown place
            {"steps": [{"destination": "dock"}]},  # no color and no shape
            {"steps": [{"color": "red", "relation": "nearest_to", "destination": "dock"}]},
            {
                "steps": [
                    {
                        "color": "red",
                        "relation": "leftmost",
                        "anchor": "shelf",
                        "destination": "dock",
                    }
                ]
            },
            {
                "steps": [
                    {"color": "red", "relation": "in_room", "anchor": "dock", "destination": "dock"}
                ]
            },
            {"steps": [{"color": "red", "destination": "dock", "extra": 1}]},  # extra key
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_program(data)

    def test_object_anchor(self):
        s = FetchStep(
            shape="box", relation="nearest_to", anchor="blue cylinder", destination="table"
        )
        self.assertEqual(s.anchor_object, ("blue", "cylinder"))

    def test_json_schema_is_valid_json(self):
        schema = llm_json_schema()
        json.dumps(schema)
        self.assertIn("steps", schema["properties"])


class _FakeLLM(LLMParser):
    """LLMParser with canned model outputs (tests the retry/reject logic offline)."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.calls = []

    def _query(self, text, error=None, previous=None):
        self.calls.append(error)
        self.messages = self._messages(text, error, previous)
        return self.outputs.pop(0)


class TestLLMParserLogic(unittest.TestCase):
    good = json.dumps({"steps": [{"color": "red", "shape": "box", "destination": "dock"}]})

    def test_valid_first_try(self):
        p = _FakeLLM([self.good])
        self.assertEqual(
            p.parse_program("bring the red box to the dock").describe(), "1. fetch(red box -> dock)"
        )
        self.assertEqual(p.last_info["attempts"], 1)

    def test_retry_once_then_accept(self):
        p = _FakeLLM(['{"steps": [{"color": "red", "destination": "moon"}]}', self.good])
        p.parse_program("bring the red box to the dock")
        self.assertEqual(p.last_info["attempts"], 2)
        self.assertIn("moon", p.calls[1])  # the error is fed back to the model
        roles = [m["role"] for m in p.messages[1:]]  # after the system prompt
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:], strict=False)))
        self.assertIn("moon", p.messages[-2]["content"])  # its previous answer

    def test_reject_after_retries(self):
        bad = '{"steps": [{"destination": "dock"}]}'
        p = _FakeLLM(["not json", bad, bad])
        with self.assertRaises(ValueError):
            p.parse_program("bring the red box to the dock")
        self.assertEqual(p.last_info["rejected"], "invalid")

    def test_model_rejection_after_three_empty_plans(self):
        p = _FakeLLM(['{"steps": []}'] * 3)
        with self.assertRaises(ValueError):
            p.parse_program("sing me a song")
        self.assertEqual(p.last_info["rejected"], "model")
        self.assertEqual(p.last_info["attempts"], 3)
        self.assertIn("empty plan", p.messages[-1]["content"])  # neutral feedback

    def test_empty_plan_is_retried(self):
        p = _FakeLLM(['{"steps": []}', self.good])
        p.parse_program("bring the red box to the dock")
        self.assertEqual(p.last_info["attempts"], 2)
        self.assertEqual(p.last_info["empty"], 1)

    def test_ungrounded_color_is_rejected(self):
        hallucinated = json.dumps(
            {"steps": [{"color": "red", "shape": "box", "destination": "dock"}]}
        )
        p = _FakeLLM([hallucinated] * 3)
        with self.assertRaises(ValueError):
            p.parse_program("bring the box to the dock")

    def test_missing_model_is_a_clear_error(self):
        p = LLMParser(LLMConfig(model_path="/nonexistent/model.gguf"))
        with self.assertRaises(RuntimeError) as cm:
            p.parse_program("bring the red box to the dock")
        self.assertIn("model file not found", str(cm.exception))


@unittest.skipUnless(
    os.path.isfile(MODEL) and os.environ.get("FETCHBOT_TEST_LLM") == "1",
    "set FETCHBOT_TEST_LLM=1 and download the GGUF model to run",
)
class TestLLMParserModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = LLMParser()

    def test_simple(self):
        prog = self.parser.parse_program("bring the red box to the dock")
        self.assertEqual(prog.describe(), "1. fetch(red box -> dock)")

    def test_multi_step(self):
        prog = self.parser.parse_program(
            "take the green ball to the table and then the blue mug to the kitchen"
        )
        self.assertEqual(len(prog.steps), 2)


if __name__ == "__main__":
    unittest.main()


class TestThinkingHelpers(unittest.TestCase):
    def test_json_part_strips_fences(self):
        from fetchbot.language.parsers import _json_part

        self.assertEqual(_json_part('```json\n{"steps": []}\n```'), '{"steps": []}')
        self.assertEqual(_json_part(""), "")

    def test_lenient_grounding(self):
        from fetchbot.language import validate_program
        from fetchbot.language.parsers import ungrounded_words

        def prog(color):
            return validate_program(
                {"steps": [{"color": color, "shape": "box", "destination": "dock"}]}
            )

        # no descriptive word at all: an invented color is still caught
        self.assertEqual(
            ungrounded_words(prog("blue"), "bring all the boxes to the dock", lenient=True),
            ["blue"],
        )
        # an unknown shade word: accepted only in lenient mode
        self.assertEqual(
            ungrounded_words(prog("red"), "bring the zorbish box to the dock", lenient=True), []
        )
        self.assertEqual(
            ungrounded_words(prog("red"), "bring the zorbish box to the dock"), ["red"]
        )
        # a known non-vocabulary color blocks it
        self.assertEqual(
            ungrounded_words(prog("red"), "bring the purple box to the dock", lenient=True), ["red"]
        )
