from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SeparationError(Exception):
    ...

class TranscriptionError(Exception):
    ...

class IdentificationError(Exception):
    ...

class AnalysisError(Exception):
    ...

class PostProcessingError(Exception):
    ...

@dataclass
class StageResult:
    status: str
    data: Any = None
    error: str | None = None
    partial: dict[str, Any] = field(default_factory=dict)
