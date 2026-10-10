from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from condsites.references import RefItem


class Condition(str, Enum):
    FULL = "full"
    SEQUENCE_ONLY = "sequence_only"
    XVECTOR_ONLY = "xvector_only"
    NEITHER = "neither"


@dataclass(frozen=True)
class MatrixProfile:
    conditions: tuple[Condition, ...]
    cells: Mapping[Condition, tuple[tuple[int, str], ...]]
    reference_counts: tuple[int, ...]
    duration_regimes: tuple[str, ...]
    reference_order_repeats: int


MATRIX_PROFILES: Mapping[str, MatrixProfile] = {
    "cosyvoice_factorial": MatrixProfile(
        conditions=tuple(Condition),
        cells={
            Condition.FULL: (
                (1, "fixed"),
                (1, "scaling"),
                (3, "fixed"),
                (3, "scaling"),
            ),
            Condition.SEQUENCE_ONLY: ((1, "fixed"),),
            Condition.XVECTOR_ONLY: (
                (1, "fixed"),
                (1, "scaling"),
                (3, "fixed"),
                (3, "scaling"),
            ),
            Condition.NEITHER: ((1, "fixed"),),
        },
        reference_counts=(1, 3),
        duration_regimes=("fixed", "scaling"),
        reference_order_repeats=2,
    ),
    "cosyvoice_nested": MatrixProfile(
        conditions=tuple(Condition),
        cells={
            Condition.FULL: (
                (1, "fixed"),
                (1, "scaling"),
                (3, "fixed"),
                (3, "scaling"),
            ),
            Condition.SEQUENCE_ONLY: ((1, "fixed"),),
            Condition.XVECTOR_ONLY: (
                (1, "fixed"),
                (1, "scaling"),
                (3, "fixed"),
                (3, "scaling"),
            ),
            Condition.NEITHER: ((1, "fixed"),),
        },
        reference_counts=(1, 3),
        duration_regimes=("fixed", "scaling"),
        reference_order_repeats=2,
    ),
    "matched_pathway_n1": MatrixProfile(
        conditions=(Condition.FULL, Condition.XVECTOR_ONLY),
        cells={
            Condition.FULL: ((1, "fixed"),),
            Condition.XVECTOR_ONLY: ((1, "fixed"),),
        },
        reference_counts=(1,),
        duration_regimes=("fixed",),
        reference_order_repeats=1,
    ),
}


@dataclass
class PreparedReference:
    condition: Condition
    refs: tuple[RefItem, ...]
    prompt_path: Path | None
    prompt_text: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    waveform: np.ndarray
    sample_rate: int
    metadata: dict[str, Any]


class TTSBackend(Protocol):
    name: str
    revision: str

    def prepare_reference(
        self,
        refs: tuple[RefItem, ...],
        condition: Condition,
        work_dir: Path,
    ) -> PreparedReference: ...

    def synthesize(
        self,
        prepared: PreparedReference,
        target_text: str,
        seed: int,
    ) -> GenerationResult: ...
