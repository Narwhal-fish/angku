"""Port: observe what a run actually runs on (code state, image, library versions, hardware).

Must be picklable: a distributed launcher ships it to the rank that trains, so the
record describes the training process rather than the driver.
"""

from __future__ import annotations

from typing import Protocol

from angku.domain.environment import RunEnvironment
from angku.domain.experiment import ExperimentSpec


class EnvironmentProbe(Protocol):
    def capture(self, spec: ExperimentSpec) -> RunEnvironment:
        """Never raises for "cannot tell": unknown facts are recorded as None."""
        ...
