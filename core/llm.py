"""
core/llm.py
───────────
LLM factory — returns the right LangChain chat model based on config.
Supports Groq (cloud, fast, free tier) and Ollama (local, offline).

Usage:
    from core.llm import get_llm
    llm = get_llm()               # uses DEFAULT_PROVIDER from .env
    llm = get_llm("groq")         # force Groq
    llm = get_llm("ollama")       # force Ollama
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_PROV  = os.getenv("DEFAULT_PROVIDER", "groq")


def get_llm(provider: str = None, temperature: float = 0.0):
    """
    Returns a LangChain chat model.

    Args:
        provider: "groq" | "ollama" | None (uses DEFAULT_PROVIDER from .env)
        temperature: 0.0 = deterministic, 1.0 = creative

    Returns:
        A LangChain BaseChatModel instance
    """
    provider = (provider or DEFAULT_PROV).lower()

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found in .env — "
                    "get a free key at https://console.groq.com"
                )
            return ChatGroq(
                model=GROQ_MODEL,
                temperature=temperature,
                api_key=api_key
            )
        except ImportError:
            raise ImportError("Run: pip install langchain-groq")

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_URL,
                temperature=temperature
            )
        except ImportError:
            raise ImportError("Run: pip install langchain-ollama")

    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'groq' or 'ollama'.")


def test_llm_connection(provider: str = None) -> bool:
    """
    Quick sanity check — returns True if the LLM responds.
    Run this once after setup to confirm everything works.
    """
    try:
        llm = get_llm(provider)
        response = llm.invoke("Say 'OK' and nothing else.")
        print(f"[LLM OK] Provider: {provider or DEFAULT_PROV} | Response: {response.content}")
        return True
    except Exception as e:
        print(f"[LLM FAIL] {e}")
        return False


if __name__ == "__main__":
    # Run: python core/llm.py
    test_llm_connection()
