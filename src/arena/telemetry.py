"""Optional trace sinks for inspecting individual evaluation decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class TraceSink(Protocol):
    """Minimal event boundary implemented by OpenTelemetry and test sinks."""

    def record(self, name: str, attributes: Mapping[str, str | int | float | bool]) -> None:
        """Record one completed evaluation operation."""
        ...


class OpenTelemetryTraceSink:
    """Export evaluation operations as OpenTelemetry spans over OTLP HTTP."""

    def __init__(self, endpoint: str, service_name: str = "agent-payment-arena") -> None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        self._provider = provider
        self._tracer = provider.get_tracer("arena.evaluation")

    def record(self, name: str, attributes: Mapping[str, str | int | float | bool]) -> None:
        """Create and immediately complete one span with stable attributes."""
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def close(self) -> None:
        """Flush queued spans before process exit."""
        self._provider.shutdown()
