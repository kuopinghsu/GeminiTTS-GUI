import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_prep import (
    group_sentences,
    parse_dialogue_turns,
    prepare_text,
    remove_inline_reference_numbers,
    replace_letter_period_sequences,
    split_sentences,
    split_text_on_pauses,
    style_for_speed,
)


class PauseTests(unittest.TestCase):
    def test_default_pause(self):
        parts = split_text_on_pauses("hello [pause] world")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][0], "hello ")
        self.assertAlmostEqual(parts[0][1], 0.5)
        self.assertEqual(parts[1][0], " world")
        self.assertEqual(parts[1][1], 0.0)

    def test_timed_pause(self):
        parts = split_text_on_pauses("a [pause:1.5] b [pause:2s] c")
        self.assertAlmostEqual(parts[0][1], 1.5)
        self.assertAlmostEqual(parts[1][1], 2.0)
        self.assertEqual(parts[2][1], 0.0)

    def test_ellipsis_stays_in_text(self):
        parts = split_text_on_pauses("wait... okay")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0], "wait... okay")


class CleaningTests(unittest.TestCase):
    def test_keep_decimals(self):
        self.assertEqual(
            remove_inline_reference_numbers("The trace is 2.521 seconds."),
            "The trace is 2.521 seconds.",
        )

    def test_strip_footnote(self):
        self.assertEqual(remove_inline_reference_numbers("See paper.3 Next."), "See paper. Next.")

    def test_initials(self):
        self.assertEqual(replace_letter_period_sequences("J.R.R. Tolkien"), "J R R Tolkien")


class ChunkTests(unittest.TestCase):
    def test_english_sentences(self):
        text = "Hello there. How are you? I am fine."
        self.assertEqual(split_sentences(text), ["Hello there.", "How are you?", "I am fine."])

    def test_cjk_sentences(self):
        text = "你好。今天天气真好！我们走吧？"
        self.assertEqual(split_sentences(text), ["你好。", "今天天气真好！", "我们走吧？"])

    def test_group_respects_max(self):
        chunks = group_sentences(["aaaa", "bbbb", "cccc"], max_chars=9, joiner=" ")
        self.assertEqual(chunks, ["aaaa bbbb", "cccc"])

    def test_prepare_chunks_long_latin(self):
        text = " ".join(["Sentence number %d is here." % i for i in range(200)])
        chunks = prepare_text(text, False, True, False, False)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 2800 for c in chunks))


class DialogueTests(unittest.TestCase):
    def test_parse_two_speakers(self):
        text = "Joe: How's it going today Jane?\nJane: Not too bad, how about you?"
        turns = parse_dialogue_turns(text)
        self.assertEqual(turns[0], ("Joe", "How's it going today Jane?"))
        self.assertEqual(turns[1][0], "Jane")

    def test_not_dialogue(self):
        self.assertIsNone(parse_dialogue_turns("Just a paragraph of narration."))

    def test_chinese_speakers(self):
        text = "小明：你好吗？\n小红：我很好。"
        turns = parse_dialogue_turns(text)
        self.assertEqual(turns[0][0], "小明")


class StyleTests(unittest.TestCase):
    def test_speed_maps_to_style(self):
        self.assertEqual(style_for_speed("", 1.0), "")
        self.assertIn("slowly", style_for_speed("warm", 0.8))
        self.assertIn("rapidly", style_for_speed("", 1.3))
        self.assertTrue(style_for_speed("cheerful", 1.2).startswith("cheerful"))


if __name__ == "__main__":
    unittest.main()
