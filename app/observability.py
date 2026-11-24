import os

from dotenv import load_dotenv

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:  # Optional until the local tracing dependencies are installed.
    LangChainInstrumentor = None
    trace = None
    OTLPSpanExporter = None
    TracerProvider = None
    BatchSpanProcessor = None


def load_settings() -> tuple[str, str]:
    """Validate local configuration before starting a worker or API process."""
    load_dotenv()
    mode = os.getenv("MODEL_MODE", "offline").lower()
    if mode not in {"offline", "openai", "nvidia", "ollama"}:
        raise ValueError("MODEL_MODE must be offline, openai, nvidia, or ollama")
    if mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required when MODEL_MODE=openai")
    if mode == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
        raise ValueError("NVIDIA_API_KEY is required when MODEL_MODE=nvidia")
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true" and not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING=true")
    os.environ.setdefault("LANGSMITH_PROJECT", "intelligence-system-final")
    return mode, os.getenv("REDIS_URL", "redis://localhost:6379/0")


def configure_phoenix_tracing():
    """Export graph and LangChain spans to the local Phoenix OTLP collector."""
    if os.getenv("PHOENIX_TRACING", "false").lower() != "true":
        return None
    if any(value is None for value in (LangChainInstrumentor, trace, OTLPSpanExporter, TracerProvider, BatchSpanProcessor)):
        raise RuntimeError("Install Phoenix tracing dependencies before enabling PHOENIX_TRACING")
    provider = TracerProvider()
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider.get_tracer("intelligence-system.worker")
