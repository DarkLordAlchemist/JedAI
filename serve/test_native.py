"""serve/native.py: the Monitor's numbers from the requests that pass through, without a model or a GPU."""
import json
import unittest

from serve.native import Stats, usage_of


class UsageTests(unittest.TestCase):
    def test_chat_completions_timings(self):
        acc = {}
        usage_of({"usage": {"prompt_tokens": 80, "completion_tokens": 20},
                  "timings": {"prompt_n": 30, "cache_n": 50, "prompt_ms": 100.0, "predicted_n": 20, "predicted_ms": 500.0}}, acc)
        self.assertEqual((acc["prompt_tokens"], acc["reused"], acc["output_tokens"]), (80, 50, 20))
        self.assertEqual((acc["prompt_ms"], acc["decode_ms"]), (100.0, 500.0))

    def test_anthropic_usage(self):
        acc = {}
        usage_of({"type": "message_start", "message": {"usage": {"input_tokens": 14, "output_tokens": 1}}}, acc)
        usage_of({"type": "message_delta", "usage": {"output_tokens": 39}}, acc)
        self.assertEqual((acc["prompt_tokens"], acc["output_tokens"]), (14, 39))

    def test_junk_is_ignored(self):
        acc = {}
        usage_of([1, 2], acc)
        usage_of({"usage": "x", "timings": 3}, acc)
        self.assertEqual(acc, {})


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.stats = Stats("gpt-oss-20b", 131072, {})
        self.stats.loaded = True
        self.addCleanup(self.stats.telemetry.close)

    def test_request_lifecycle(self):
        self.assertEqual(self.stats.metrics()["live"]["state"], "idle")
        rid = self.stats.begin("/v1/messages")
        self.assertEqual(self.stats.metrics()["live"]["state"], "reading")
        self.stats.token(rid, 3)
        self.assertEqual(self.stats.metrics()["live"]["state"], "generating")
        self.stats.end(rid, "stop", {"prompt_tokens": 10, "output_tokens": 3, "decode_ms": 100.0})
        m = self.stats.metrics()
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["decode_tok_s"], 30.0)
        self.assertEqual(m["totals"]["requests"], 1)
        json.dumps(m)

    def test_a_whole_answer_has_no_made_up_speed(self):
        rid = self.stats.begin("/v1/messages")
        self.stats.token(rid, 40)           # a non-streaming answer arrives in one piece
        self.stats.end(rid, "stop", {"prompt_tokens": 10, "output_tokens": 40})
        self.assertIsNone(self.stats.metrics()["requests"][0]["decode_tok_s"])

    def test_unloaded_until_ready(self):
        self.stats.loaded = False
        self.assertEqual(self.stats.metrics()["live"]["state"], "unloaded")


if __name__ == "__main__":
    unittest.main()
