"""Registries for researcher-defined components (models, losses, augmentations, datasets).

The platform never imports a concrete model. Researchers register factories
under a name; ``experiment.yaml`` refers to that name and supplies ``params``.
Each registration carries a Pydantic ``Params`` schema so that a typo inside
``model.params`` fails at build time with a clear message instead of being
silently ignored.

This module depends on Pydantic only. Concrete factories (which import torch /
MONAI) live in the ``experiments`` package and are loaded via ``discover``.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T")
P = TypeVar("P", bound="Params")


class Params(BaseModel):
    """Base class for a component's parameter schema.

    Unknown keys are rejected, values are immutable. Subclass and declare the
    fields the component accepts (with defaults where sensible).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class Entry(Generic[T]):
    name: str
    factory: Callable[[Any], T]
    params_schema: type[Params]


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}

    # -- registration --------------------------------------------------------

    def register(
        self, name: str, *, params: type[Params]
    ) -> Callable[[Callable[[Any], T]], Callable[[Any], T]]:
        """Decorator: ``@MODELS.register("swinunetr", params=SwinUNETRParams)``.

        The decorated factory receives one argument: a validated ``params`` instance.
        """
        if not isinstance(params, type) or not issubclass(params, Params):
            raise RegistryError(
                f"{self.kind} {name!r}: params must be a subclass of mlops.registry.Params"
            )

        def deco(factory: Callable[[Any], T]) -> Callable[[Any], T]:
            if name in self._entries:
                existing = self._entries[name].factory
                raise RegistryError(
                    f"{self.kind} {name!r} already registered by "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
            self._entries[name] = Entry(name=name, factory=factory, params_schema=params)
            return factory

        return deco

    # -- lookup --------------------------------------------------------------

    def get(self, name: str) -> Entry[T]:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "<none>"
            raise RegistryError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    # -- construction --------------------------------------------------------

    def validate_params(self, name: str, params: dict[str, Any]) -> Params:
        entry = self.get(name)
        try:
            return entry.params_schema.model_validate(params)
        except ValidationError as e:
            raise RegistryError(f"invalid params for {self.kind} {name!r}:\n{e}") from e

    def build(self, name: str, params: dict[str, Any]) -> T:
        entry = self.get(name)
        return entry.factory(self.validate_params(name, params))


MODELS: Registry[Any] = Registry("model")
LOSSES: Registry[Any] = Registry("loss")
AUGMENTATIONS: Registry[Any] = Registry("augmentation")
# Data providers, keyed by ``dataset.name``. Unlike the other kinds, ``params`` here is the
# provider's *operational* config (e.g. ``root``: where the files live on this machine), which
# never enters any id; the yaml's ``preprocessing`` section goes to ``DataProvider.load``.
DATASETS: Registry[Any] = Registry("dataset")


def discover(package: str) -> list[str]:
    """Import every submodule of ``package`` so their ``@register`` calls run.

    Returns the imported module names. Idempotent: modules already imported are
    not re-executed, so registrations happen exactly once.
    """
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        return [pkg.__name__]
    imported: list[str] = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(info.name)
        imported.append(info.name)
    return imported
