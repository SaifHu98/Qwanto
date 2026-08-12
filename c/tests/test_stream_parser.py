import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import openai_server

class TestStreamParser(unittest.TestCase):

    def setUp(self):
        # Mock param types
        self.param_types = {
            "get_weather": {
                "location": "string",
                "days": "integer"
            }
        }
        
    def _feed(self, parser, text, chunk_size=None):
        deltas = []
        if chunk_size:
            for i in range(0, len(text), chunk_size):
                deltas.extend(parser.add_chunk(text[i:i+chunk_size]))
        else:
            deltas.extend(parser.add_chunk(text))
        deltas.extend(parser.finalize())
        return deltas

    def test_text_only(self):
        parser = openai_server.StreamParser({})
        deltas = self._feed(parser, "Hello world!")
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["content"], "Hello world!")

    def test_text_chunked(self):
        parser = openai_server.StreamParser({})
        deltas = self._feed(parser, "Hello world!", chunk_size=1)
        # It buffers max lookahead, so there will be some chunks
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(content, "Hello world!")

    def test_thinking(self):
        parser = openai_server.StreamParser({})
        text = "<think>Wait, I should check the weather.</think> Let me do that."
        deltas = self._feed(parser, text)
        
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        
        self.assertEqual(reasoning, "Wait, I should check the weather.")
        self.assertEqual(content, " Let me do that.")

    def test_thinking_chunked(self):
        parser = openai_server.StreamParser({})
        text = "<think>Wait, I should check the weather.</think> Let me do that."
        deltas = self._feed(parser, text, chunk_size=1)
        
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        
        self.assertEqual(reasoning, "Wait, I should check the weather.")
        self.assertEqual(content, " Let me do that.")

    def test_single_tool(self):
        parser = openai_server.StreamParser(self.param_types)
        text = "I'll check the weather.<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>San Francisco</arg_value>\n<arg_key>days</arg_key><arg_value>3</arg_value>\n</tool_call>"
        deltas = self._feed(parser, text)
        
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(content, "I'll check the weather.")
        
        tool_args = ""
        tc_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        
        # Check first delta has name and ID
        self.assertEqual(tc_deltas[0]["function"]["name"], "get_weather")
        self.assertEqual(tc_deltas[0]["function"]["arguments"], "{")
        
        for tcd in tc_deltas[1:]:
            tool_args += tcd["function"]["arguments"]
            
        # The arguments should eventually evaluate to {"location": "San Francisco", "days": 3}
        self.assertEqual(tool_args, "\"location\": \"San Francisco\", \"days\": 3}")

    def test_single_tool_chunked(self):
        parser = openai_server.StreamParser(self.param_types)
        text = "I'll check the weather.<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>San Francisco</arg_value>\n<arg_key>days</arg_key><arg_value>3</arg_value>\n</tool_call>"
        deltas = self._feed(parser, text, chunk_size=1)
        
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(content, "I'll check the weather.")
        
        tool_args = ""
        tc_deltas = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
        
        for tcd in tc_deltas:
            if "arguments" in tcd["function"]:
                tool_args += tcd["function"]["arguments"]
                
        self.assertTrue(tool_args.startswith("{"))
        self.assertTrue(tool_args.endswith("}"))
        self.assertIn('"location": "San Francisco"', tool_args)
        self.assertIn('"days": 3', tool_args)

    def test_multiple_tools(self):
        parser = openai_server.StreamParser(self.param_types)
        text = "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>Paris</arg_value>\n</tool_call> And <tool_call>get_weather\n<arg_key>location</arg_key><arg_value>London</arg_value>\n</tool_call>"
        deltas = self._feed(parser, text, chunk_size=2)
        
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(content, " And ")
        
        tc0_args = ""
        tc1_args = ""
        for d in deltas:
            if "tool_calls" in d:
                tc = d["tool_calls"][0]
                if tc["index"] == 0 and "arguments" in tc["function"]:
                    tc0_args += tc["function"]["arguments"]
                elif tc["index"] == 1 and "arguments" in tc["function"]:
                    tc1_args += tc["function"]["arguments"]
                    
        self.assertEqual(tc0_args, "{\"location\": \"Paris\"}")
        self.assertEqual(tc1_args, "{\"location\": \"London\"}")

    def test_escaped_quotes_in_args(self):
        parser = openai_server.StreamParser(self.param_types)
        text = "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>San \"Franny\" Francisco</arg_value>\n</tool_call>"
        deltas = self._feed(parser, text)
        
        tool_args = ""
        for d in deltas:
            if "tool_calls" in d:
                tool_args += d["tool_calls"][0]["function"].get("arguments", "")
                
        self.assertEqual(tool_args, "{\"location\": \"San \\\"Franny\\\" Francisco\"}")

    def test_malformed_interruption(self):
        parser = openai_server.StreamParser(self.param_types)
        text = "<tool_call>get_weather\n<arg_key>location</arg_key><arg_value>San "
        deltas = self._feed(parser, text)
        
        tool_args = ""
        for d in deltas:
            if "tool_calls" in d:
                tool_args += d["tool_calls"][0]["function"].get("arguments", "")
                
        self.assertEqual(tool_args, "{\"location\": \"San \"}")

if __name__ == '__main__':
    unittest.main()
