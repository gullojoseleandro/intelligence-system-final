import os

from dotenv import load_dotenv


def load_settings() -> tuple[str, str]:
    """Validate local configuration before starting a worker or API process."""
    load_dotenv()
    mode = os.getenv("MODEL_MODE", "offline").lower()
    if mode not in {"offline", "openai", "nvidia"}:
        raise ValueError("MODEL_MODE must be offline, openai, or nvidia")
    if mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required when MODEL_MODE=openai")
    if mode == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is required when MODEL_MODE=nvidia")
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING=true")
    os.environ.setdefault("LANGSMITH_PROJECT", "coderhouse-preentrega-7")
    return mode, os.getenv("REDIS_URL", "redis://localhost:6379/0")
