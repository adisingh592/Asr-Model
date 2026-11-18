"""
Unified Voice Assistant Brain powered by Groq Cloud API (Free Tier).
Supports ultra-fast Llama-3.1-8b-instant with function calling for live weather, time, and system info.
Requires ZERO external packages (uses Python standard library urllib.request).
"""

from datetime import datetime
import json
import os
import platform
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_assistant.weather_tool import get_live_weather

def _load_env_file():
    """Auto-loads .env file from workspace root using Python standard library."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env"
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ and v:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an ultra-fast, helpful voice assistant for an ESP32 smart edge system. "
    "The user is listening to your answer spoken aloud through a speech synthesizer. "
    "Keep every response conversational, direct, and under 2 sentences maximum. "
    "Do NOT use markdown, asterisks, bullet points, numbered lists, emojis, or code snippets."
)

# Tool Definitions for Groq Function Calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get real-time live weather conditions, temperature, and humidity for a specified city or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. 'Delhi', 'London', 'New York'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time, day, and date.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Get local host computer information including OS and current timestamp.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def execute_local_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Executes local tool functions called by Groq."""
    if tool_name == "get_current_weather":
        loc = arguments.get("location", "Delhi")
        return get_live_weather(loc)
    elif tool_name == "get_current_time":
        now = datetime.now()
        return f"Current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
    elif tool_name == "get_system_status":
        return f"System is running {platform.system()} {platform.release()} with normal status at {datetime.now().strftime('%H:%M:%S')}."
    return f"Tool {tool_name} is not recognized."


class GroqVoiceBrain:
    """Unified Voice Intelligence Engine connecting STT text to Groq Cloud."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model

    def query(self, user_text: str) -> Dict[str, Any]:
        """
        Sends the user text query to Groq with tool-calling capabilities.
        
        Returns:
            dict with:
                - response_text: str (the final voice response)
                - tool_used: Optional[str]
                - latency_ms: float
        """
        t0 = time.perf_counter()
        user_clean = user_text.strip()
        if not user_clean:
            return {
                "response_text": "I didn't catch that. Could you please repeat your question?",
                "tool_used": None,
                "latency_ms": 0.0
            }

        # Fallback if no GROQ_API_KEY is configured
        if not self.api_key:
            return self._offline_tool_router(user_clean, t0)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ESP32-Voice-Assistant/1.0"
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_clean}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
            "max_tokens": 150,
            "temperature": 0.4
        }

        tool_used = None
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(GROQ_ENDPOINT, data=req_data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            choice = result["choices"][0]
            message = choice["message"]

            # Handle Tool Calls
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                fn_name = tool_call["function"]["name"]
                fn_args_raw = tool_call["function"].get("arguments", "{}")
                try:
                    fn_args = json.loads(fn_args_raw)
                except Exception:
                    fn_args = {}

                tool_used = fn_name
                tool_result_str = execute_local_tool(fn_name, fn_args)

                # Send tool output back to Groq for natural spoken synthesis
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result_str
                })

                follow_up_payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.3
                }

                follow_up_data = json.dumps(follow_up_payload).encode("utf-8")
                follow_up_req = urllib.request.Request(GROQ_ENDPOINT, data=follow_up_data, headers=headers, method="POST")

                with urllib.request.urlopen(follow_up_req, timeout=8) as f_resp:
                    follow_up_res = json.loads(f_resp.read().decode("utf-8"))
                    final_text = follow_up_res["choices"][0]["message"]["content"].strip()
            else:
                final_text = message.get("content", "").strip()

            t1 = time.perf_counter()
            return {
                "response_text": final_text,
                "tool_used": tool_used,
                "latency_ms": (t1 - t0) * 1000.0
            }

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else str(e)
            print(f"[Groq Client] HTTP error {e.code}: {err_body}")
            # Graceful fallback to offline router
            return self._offline_tool_router(user_clean, t0, warning="Groq API key error. Using direct tools.")
        except Exception as e:
            print(f"[Groq Client] Network error: {e}")
            return self._offline_tool_router(user_clean, t0, warning="Network unreachable. Using direct tools.")

    def _offline_tool_router(self, user_text: str, t0: float, warning: Optional[str] = None) -> Dict[str, Any]:
        """Direct local/web tool router when GROQ_API_KEY is not provided."""
        text_lower = user_text.lower()
        tool_used = None

        if "weather" in text_lower or "temperature" in text_lower or "rain" in text_lower:
            tool_used = "get_current_weather"
            # Extract possible city name
            words = text_lower.replace("?", "").replace(".", "").split()
            city = "Delhi"
            if "in" in words:
                idx = words.index("in")
                if idx + 1 < len(words):
                    city = " ".join(words[idx + 1:])
            response = get_live_weather(city)
        elif "time" in text_lower or "clock" in text_lower:
            tool_used = "get_current_time"
            now = datetime.now()
            response = f"The time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}."
        elif "date" in text_lower or "day" in text_lower or "today" in text_lower:
            tool_used = "get_current_time"
            now = datetime.now()
            response = f"Today is {now.strftime('%A, %B %d, %Y')}."
        elif "status" in text_lower or "system" in text_lower:
            tool_used = "get_system_status"
            response = f"All systems are operational on {platform.system()}."
        else:
            response = (
                f"I heard you say: '{user_text}'. "
                f"To ask any general question, set your free GROQ_API_KEY in your environment."
            )

        t1 = time.perf_counter()
        return {
            "response_text": response,
            "tool_used": tool_used,
            "latency_ms": (t1 - t0) * 1000.0,
            "note": warning
        }


if __name__ == "__main__":
    brain = GroqVoiceBrain()
    print("Testing Groq Brain (Weather):")
    res = brain.query("What is the weather in Delhi?")
    print(res)
    print("\nTesting Groq Brain (Time):")
    res2 = brain.query("What time is it right now?")
    print(res2)
