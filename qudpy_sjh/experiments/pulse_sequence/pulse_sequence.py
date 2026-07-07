"""通用 multi-pulse / pulse-sequence single-run scaffold。

本模块只描述一次给定 field configuration 的构造：

    PulseSpec / FieldGroupSpec / PulseSequenceSpec -> FieldPhySeries

它不调用 solver，不执行 phase cycling，不包含 TA / 2DES 专用语义。
`FieldPhyRoot` 仍是通用抽象接口；`CarrierEnvelopeField` 是当前
phase-aware workflow 的首个正式支持 backend。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np

from qudpy_sjh.utils.fields import FieldPhyRoot, FieldPhySeries
from qudpy_sjh.utils.fields.carrier_envelope import CarrierEnvelopeField


def validate_phase_tag(tag: str | None, *, allow_none: bool = True) -> str | None:
    """校验 phase tag；None 表示该 pulse 不参与 phase cycling。"""

    if tag is None:
        if allow_none:
            return None
        raise ValueError("phase_tag must not be None.")
    text = str(tag).strip()
    if not text:
        raise ValueError("phase_tag must not be empty or blank.")
    return text


def validate_pulse_name(name: str) -> str:
    """校验用于 FieldPhySeries.sub_field_names 的 pulse 名称。"""

    text = str(name).strip()
    if not text:
        raise ValueError("pulse name must not be empty or blank.")
    return text


def _stable_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _check_duplicate_names(names: Sequence[str], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"{label} must be unique. Duplicates: {duplicates}")


def normalize_phase_vector(
    phase_vector: Mapping[str, float] | None,
    *,
    known_tags: Sequence[str] | None = None,
    fill_missing_with_zero: bool = True,
) -> dict[str, float]:
    """标准化 phase vector，并对未知 phase tag fail-fast。"""

    data = (
        {}
        if phase_vector is None
        else {
            validate_phase_tag(key, allow_none=False): float(value)
            for key, value in phase_vector.items()
        }
    )
    if known_tags is None:
        return data

    tags = tuple(validate_phase_tag(tag, allow_none=False) for tag in known_tags)
    _check_duplicate_names(tags, label="known_tags")
    known = set(tags)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"phase_vector contains unknown phase tags: {unknown}")
    if fill_missing_with_zero:
        for tag in tags:
            data.setdefault(tag, 0.0)
    return data


def is_supported_phase_backend(field: FieldPhyRoot) -> bool:
    """当前 pulse-sequence phase-aware workflow 正式支持的 field backend。"""

    return isinstance(field, CarrierEnvelopeField)


def supports_phase_override(field: FieldPhyRoot) -> bool:
    """当前 field 是否具有正式 phase override API。"""

    return callable(getattr(field, "with_phase", None)) or callable(getattr(field, "phase_shifted", None))


def _apply_phase_override(
    field: FieldPhyRoot,
    *,
    phase_rad: float,
    strict_phase_override: bool,
) -> tuple[FieldPhyRoot, bool, str]:
    """对支持的 field backend 应用 absolute phase override。"""

    phase = float(phase_rad)
    if not np.isfinite(phase):
        raise ValueError("phase_rad must be finite.")
    with_phase = getattr(field, "with_phase", None)
    if callable(with_phase):
        shifted = with_phase(phase)
        if not isinstance(shifted, FieldPhyRoot):
            raise TypeError("field.with_phase(...) must return a FieldPhyRoot instance.")
        return shifted, True, "phase override applied through field.with_phase"

    if callable(getattr(field, "phase_shifted", None)) and strict_phase_override and abs(phase) > 1.0e-15:
        raise TypeError(
            "Absolute phase override requires with_phase(...); phase_shifted(...) alone is not sufficient."
        )

    if strict_phase_override and abs(phase) > 1.0e-15:
        backend = field.__class__.__name__
        raise TypeError(
            f"Non-zero phase override requires a field backend with with_phase(...). "
            f"Got unsupported backend: {backend}."
        )

    if is_supported_phase_backend(field):
        raise TypeError(
            f"{field.__class__.__name__} is a supported phase backend but does not expose with_phase(...)."
        )
    return field, False, "field backend has no with_phase API; zero phase kept metadata-only"


def _field_payload(field: FieldPhyRoot) -> dict[str, Any]:
    if hasattr(field, "to_dict") and callable(field.to_dict):
        return field.to_dict()
    return {"class": field.__class__.__name__, "repr": repr(field), "rebuildable": False}


def _merge_metadata(base: Mapping[str, Any] | None, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(extra or {}))
    return merged


def _effective_phase(
    *,
    base_phase_rad: float,
    phase_tag: str | None,
    phase_vector: Mapping[str, float] | None,
) -> float:
    if phase_tag is None:
        normalize_phase_vector(phase_vector, known_tags=(), fill_missing_with_zero=False)
        return float(base_phase_rad)
    phases = normalize_phase_vector(phase_vector, known_tags=(phase_tag,), fill_missing_with_zero=True)
    return float(base_phase_rad) + float(phases[phase_tag])


@dataclass(frozen=True)
class PulseSpec:
    """一个 physical pulse 的模板和 phase metadata。"""

    name: str
    field_template: FieldPhyRoot
    template_center_fs: float = 0.0
    phase_tag: str | None = None
    phase_rad: float = 0.0
    independent_phase: bool = False
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_pulse_name(self.name))
        if not isinstance(self.field_template, FieldPhyRoot):
            raise TypeError("field_template must be a FieldPhyRoot instance.")
        tag = validate_phase_tag(self.phase_tag, allow_none=True)
        if bool(self.independent_phase) and tag is None:
            raise ValueError("independent_phase=True requires a non-empty phase_tag.")
        if not np.isfinite(float(self.template_center_fs)):
            raise ValueError("template_center_fs must be finite.")
        if not np.isfinite(float(self.phase_rad)):
            raise ValueError("phase_rad must be finite.")
        object.__setattr__(self, "phase_tag", tag)
        object.__setattr__(self, "template_center_fs", float(self.template_center_fs))
        object.__setattr__(self, "phase_rad", float(self.phase_rad))
        object.__setattr__(self, "independent_phase", bool(self.independent_phase))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def phase_tags(self, *, independent_only: bool = False) -> tuple[str, ...]:
        if self.phase_tag is None:
            return ()
        if independent_only and not self.independent_phase:
            return ()
        return (self.phase_tag,)

    def shifted(
        self,
        *,
        center_fs: float,
        phase_vector: Mapping[str, float] | None = None,
        phase_rad: float | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_phase_override: bool = True,
    ) -> FieldPhyRoot:
        """构造一次具体 pulse field。

        对支持 `with_phase` 的 backend，`phase_rad` 会真实改变 field 相位。
        对不支持的 backend，默认 strict 行为会拒绝非零 phase override。
        """

        requested_center = float(center_fs)
        if not np.isfinite(requested_center):
            raise ValueError("center_fs must be finite.")
        base_phase = self.phase_rad if phase_rad is None else float(phase_rad)
        if not np.isfinite(base_phase):
            raise ValueError("phase_rad must be finite.")
        effective_phase = _effective_phase(
            base_phase_rad=base_phase,
            phase_tag=self.phase_tag,
            phase_vector=phase_vector,
        )
        time_shift = requested_center - float(self.template_center_fs)
        output_name = validate_pulse_name(name) if name is not None else self.name
        pulse_metadata = _merge_metadata(
            self.metadata,
            {
                "role": "pulse",
                "pulse_name": self.name,
                "phase_tag": self.phase_tag,
                "phase_rad": effective_phase,
                "base_phase_rad": base_phase,
                "requested_center_fs": requested_center,
                "template_center_fs": float(self.template_center_fs),
                "time_shift_fs": float(time_shift),
                "independent_phase": bool(self.independent_phase),
                "phase_override_applied": None,
                "phase_override_note": None,
            },
        )
        pulse_metadata.update(dict(metadata or {}))
        phased_template, applied, note = _apply_phase_override(
            self.field_template,
            phase_rad=effective_phase,
            strict_phase_override=bool(strict_phase_override),
        )
        pulse_metadata["phase_override_applied"] = applied
        pulse_metadata["phase_override_note"] = note
        return phased_template.time_shifted(
            time_shift,
            name=output_name,
            metadata=pulse_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "template_center_fs": float(self.template_center_fs),
            "phase_tag": self.phase_tag,
            "phase_rad": float(self.phase_rad),
            "independent_phase": bool(self.independent_phase),
            "field_template": _field_payload(self.field_template),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FieldGroupSpec:
    """多个 PulseSpec 组成的 coherent physical field group。"""

    name: str
    pulses: tuple[PulseSpec, ...]
    phase_tag: str | None = None
    phase_rad: float = 0.0
    independent_phase: bool = False
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_pulse_name(self.name))
        pulses = tuple(self.pulses)
        if not pulses:
            raise ValueError("FieldGroupSpec requires at least one PulseSpec.")
        for pulse in pulses:
            if not isinstance(pulse, PulseSpec):
                raise TypeError("FieldGroupSpec.pulses must contain PulseSpec instances.")
        _check_duplicate_names(tuple(pulse.name for pulse in pulses), label="FieldGroupSpec pulse names")
        tag = validate_phase_tag(self.phase_tag, allow_none=True)
        if bool(self.independent_phase) and tag is None:
            raise ValueError("independent_phase=True requires a non-empty group phase_tag.")
        if tag is not None:
            internal_tags = [pulse.phase_tag for pulse in pulses if pulse.phase_tag is not None]
            if internal_tags:
                raise ValueError(
                    "FieldGroupSpec.phase_tag takes priority; pulses inside a group-level phase_tag "
                    f"must not expose their own phase_tag. Internal tags: {internal_tags}"
                )
        if not np.isfinite(float(self.phase_rad)):
            raise ValueError("phase_rad must be finite.")
        object.__setattr__(self, "pulses", pulses)
        object.__setattr__(self, "phase_tag", tag)
        object.__setattr__(self, "phase_rad", float(self.phase_rad))
        object.__setattr__(self, "independent_phase", bool(self.independent_phase))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def phase_tags(self, *, independent_only: bool = False) -> tuple[str, ...]:
        if self.phase_tag is not None:
            if independent_only and not self.independent_phase:
                return ()
            return (self.phase_tag,)
        tags: list[str] = []
        for pulse in self.pulses:
            tags.extend(pulse.phase_tags(independent_only=independent_only))
        return _stable_unique(tags)

    def build(
        self,
        *,
        centers_fs: Mapping[str, float],
        phase_vector: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_phase_override: bool = True,
    ) -> FieldPhySeries:
        pulse_names = tuple(pulse.name for pulse in self.pulses)
        missing = [name for name in pulse_names if name not in centers_fs]
        if missing:
            raise ValueError(f"centers_fs is missing pulse centers for: {missing}")
        extra = sorted(set(centers_fs) - set(pulse_names))
        if extra:
            raise ValueError(f"centers_fs contains unknown pulse names for this group: {extra}")

        group_phase_vector: dict[str, float] = {}
        group_phase = float(self.phase_rad)
        if self.phase_tag is not None:
            group_phase_vector = normalize_phase_vector(
                phase_vector,
                known_tags=(self.phase_tag,),
                fill_missing_with_zero=True,
            )
            group_phase += float(group_phase_vector[self.phase_tag])
        else:
            known_tags = self.phase_tags()
            group_phase_vector = normalize_phase_vector(
                phase_vector,
                known_tags=known_tags,
                fill_missing_with_zero=True,
            )

        fields: list[FieldPhyRoot] = []
        for pulse in self.pulses:
            if self.phase_tag is None:
                pulse_vector = {tag: group_phase_vector[tag] for tag in pulse.phase_tags() if tag in group_phase_vector}
                pulse_phase = None
            else:
                pulse_vector = None
                pulse_phase = pulse.phase_rad + group_phase
            fields.append(
                pulse.shifted(
                    center_fs=float(centers_fs[pulse.name]),
                    phase_vector=pulse_vector,
                    phase_rad=pulse_phase,
                    metadata={
                        "field_group_name": self.name,
                        "field_group_phase_tag": self.phase_tag,
                    },
                    strict_phase_override=strict_phase_override,
                )
            )

        group_metadata = _merge_metadata(
            self.metadata,
            {
                "role": "field_group",
                "group_name": self.name,
                "phase_tag": self.phase_tag,
                "phase_rad": group_phase,
                "base_phase_rad": float(self.phase_rad),
                "independent_phase": bool(self.independent_phase),
                "phase_convention": "one coherent physical group shares one group phase_tag when provided",
                "pulses": [pulse.to_dict() for pulse in self.pulses],
            },
        )
        group_metadata.update(dict(metadata or {}))
        return FieldPhySeries(
            fields=tuple(fields),
            sub_field_names=pulse_names,
            name=self.name,
            metadata=group_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "phase_tag": self.phase_tag,
            "phase_rad": float(self.phase_rad),
            "independent_phase": bool(self.independent_phase),
            "pulses": [pulse.to_dict() for pulse in self.pulses],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PulseSequenceSpec:
    """任意数量 pulse / field group 的 single-run field scaffold。"""

    name: str
    pulses: tuple[PulseSpec, ...] = ()
    field_groups: tuple[FieldGroupSpec, ...] = ()
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_pulse_name(self.name))
        pulses = tuple(self.pulses)
        field_groups = tuple(self.field_groups)
        if not pulses and not field_groups:
            raise ValueError("PulseSequenceSpec requires at least one pulse or field_group.")
        for pulse in pulses:
            if not isinstance(pulse, PulseSpec):
                raise TypeError("PulseSequenceSpec.pulses must contain PulseSpec instances.")
        for group in field_groups:
            if not isinstance(group, FieldGroupSpec):
                raise TypeError("PulseSequenceSpec.field_groups must contain FieldGroupSpec instances.")

        all_pulse_names = [pulse.name for pulse in pulses]
        for group in field_groups:
            all_pulse_names.extend(pulse.name for pulse in group.pulses)
        _check_duplicate_names(tuple(all_pulse_names), label="PulseSequenceSpec pulse names")
        _check_duplicate_names(tuple([pulse.name for pulse in pulses] + [group.name for group in field_groups]), label="PulseSequenceSpec top-level field names")
        object.__setattr__(self, "pulses", pulses)
        object.__setattr__(self, "field_groups", field_groups)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def phase_tags(self, *, independent_only: bool = False) -> tuple[str, ...]:
        tags: list[str] = []
        for pulse in self.pulses:
            tags.extend(pulse.phase_tags(independent_only=independent_only))
        for group in self.field_groups:
            tags.extend(group.phase_tags(independent_only=independent_only))
        return _stable_unique(tags)

    def _pulse_names(self) -> tuple[str, ...]:
        names = [pulse.name for pulse in self.pulses]
        for group in self.field_groups:
            names.extend(pulse.name for pulse in group.pulses)
        return tuple(names)

    def build_field(
        self,
        *,
        centers_fs: Mapping[str, float],
        phase_vector: Mapping[str, float] | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        strict_phase_override: bool = True,
    ) -> FieldPhySeries:
        pulse_names = self._pulse_names()
        missing = [pulse_name for pulse_name in pulse_names if pulse_name not in centers_fs]
        if missing:
            raise ValueError(f"centers_fs is missing pulse centers for: {missing}")
        extra = sorted(set(centers_fs) - set(pulse_names))
        if extra:
            raise ValueError(f"centers_fs contains unknown pulse names: {extra}")

        phases = normalize_phase_vector(
            phase_vector,
            known_tags=self.phase_tags(),
            fill_missing_with_zero=True,
        )
        fields: list[FieldPhyRoot] = []
        names: list[str] = []
        for pulse in self.pulses:
            pulse_vector = {tag: phases[tag] for tag in pulse.phase_tags() if tag in phases}
            fields.append(
                pulse.shifted(
                    center_fs=float(centers_fs[pulse.name]),
                    phase_vector=pulse_vector,
                    strict_phase_override=strict_phase_override,
                )
            )
            names.append(pulse.name)
        for group in self.field_groups:
            group_vector = {tag: phases[tag] for tag in group.phase_tags() if tag in phases}
            group_centers = {pulse.name: float(centers_fs[pulse.name]) for pulse in group.pulses}
            fields.append(
                group.build(
                    centers_fs=group_centers,
                    phase_vector=group_vector,
                    strict_phase_override=strict_phase_override,
                )
            )
            names.append(group.name)

        sequence_metadata = _merge_metadata(
            self.metadata,
            {
                "role": "pulse_sequence",
                "sequence_name": self.name,
                "phase_tags": list(self.phase_tags()),
                "phase_vector": phases,
                "centers_fs": {pulse_name: float(centers_fs[pulse_name]) for pulse_name in pulse_names},
                "phase_tag_semantics": "phase_tag belongs to PulseSpec or FieldGroupSpec, not to carrier components by default",
            },
        )
        sequence_metadata.update(dict(metadata or {}))
        return FieldPhySeries(
            fields=tuple(fields),
            sub_field_names=tuple(names),
            name=validate_pulse_name(name) if name is not None else self.name,
            metadata=sequence_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "pulses": [pulse.to_dict() for pulse in self.pulses],
            "field_groups": [group.to_dict() for group in self.field_groups],
            "phase_tags": list(self.phase_tags()),
            "independent_phase_tags": list(self.phase_tags(independent_only=True)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SingleRunFieldPlan:
    """一次 concrete field configuration 的轻量 plan，不调用 run_case。"""

    sequence: PulseSequenceSpec
    centers_fs: dict[str, float]
    phase_vector: dict[str, float]
    case_name: str
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, PulseSequenceSpec):
            raise TypeError("sequence must be a PulseSequenceSpec instance.")
        object.__setattr__(self, "case_name", validate_pulse_name(self.case_name))
        centers = {validate_pulse_name(name): float(value) for name, value in self.centers_fs.items()}
        for value in centers.values():
            if not np.isfinite(value):
                raise ValueError("centers_fs values must be finite.")
        phases = normalize_phase_vector(
            self.phase_vector,
            known_tags=self.sequence.phase_tags(),
            fill_missing_with_zero=True,
        )
        object.__setattr__(self, "centers_fs", centers)
        object.__setattr__(self, "phase_vector", phases)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def build_field(self) -> FieldPhySeries:
        return self.sequence.build_field(
            centers_fs=self.centers_fs,
            phase_vector=self.phase_vector,
            name=self.case_name,
            metadata={"single_run_case_name": self.case_name, **self.metadata},
            strict_phase_override=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "sequence": self.sequence.to_dict(),
            "centers_fs": dict(self.centers_fs),
            "phase_vector": dict(self.phase_vector),
            "case_name": self.case_name,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "FieldGroupSpec",
    "PulseSequenceSpec",
    "PulseSpec",
    "SingleRunFieldPlan",
    "is_supported_phase_backend",
    "normalize_phase_vector",
    "supports_phase_override",
    "validate_phase_tag",
    "validate_pulse_name",
]
