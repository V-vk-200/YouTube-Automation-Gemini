"""Provider-neutral text client for the scripting half of the pipeline."""
import json
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["gemini", "groq"]
DEFAULT_MODELS: dict[Provider, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
}


def select_provider() -> Provider:
    """Prefer Gemini when both supported keys are configured."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    raise RuntimeError(
        "No writing-model key found. Add GEMINI_API_KEY or GROQ_API_KEY "
        "to scripting/.env (or set it in your environment)."
    )


class ScriptClient:
    """Text-generation adapter using Gemini or Groq."""
    def __init__(self, provider: Provider | None = None, model: str | None = None):
        self.provider = provider or select_provider()
        self.model = model or os.getenv(f"{self.provider.upper()}_MODEL") or DEFAULT_MODELS[self.provider]
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            from groq import Groq
            self.client = Groq()

    def complete(self, prompt: str, max_tokens: int = 7000, temperature: float = 0) -> str:
        if self.provider == "gemini":
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            content = response.text or ""
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError(f"{self.provider} returned an empty response")
        return content


# Compatibility names used by existing imports and stage CLI.
GroqClient = ScriptClient
Claude = ScriptClient


def parse_jsonish(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in [text.find("["), text.find("{")] if i >= 0], default=-1)
        end = max(text.rfind("]"), text.rfind("}"))
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"{text[:40]}... was not JSON; preserve raw output and inspect the template's required structure")
