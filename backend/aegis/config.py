"""Node configuration.

Every knob an edge deployment needs, resolvable from env (``AEGIS_*``) so the
same image runs on a kiosk, a robot and a laptop without a rebuild.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _env(key: str, default: Any) -> Any:
    raw = os.environ.get("AEGIS_" + key.upper())
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(slots=True)
class MemoryConfig:
    dim: int = field(default_factory=lambda: _env("dim", 384))
    sparse_dim: int = field(default_factory=lambda: _env("sparse_dim", 1 << 18))
    collections: tuple[str, ...] = ("episodic", "semantic", "procedural", "sensor")
    hot_capacity: int = field(default_factory=lambda: _env("hot_capacity", 20_000))
    warm_capacity: int = field(default_factory=lambda: _env("warm_capacity", 80_000))
    compaction_interval_s: float = field(default_factory=lambda: _env("compaction_interval_s", 20.0))
    consolidation_interval_s: float = field(default_factory=lambda: _env("consolidation_interval_s", 90.0))
    consolidation_threshold: float = field(default_factory=lambda: _env("consolidation_threshold", 0.94))


@dataclass(slots=True)
class InferenceConfig:
    embedder: str = field(default_factory=lambda: _env("embedder", "bge-small-en-v1.5"))
    reranker: str = field(default_factory=lambda: _env("reranker", "cross-encoder-mini"))
    precision: str = field(default_factory=lambda: _env("precision", "int8-dynamic"))
    batch_window_ms: float = field(default_factory=lambda: _env("batch_window_ms", 8.0))
    max_batch: int = field(default_factory=lambda: _env("max_batch", 32))
    model_dir: str = field(default_factory=lambda: _env("model_dir", "models"))
    thermal_ceiling_c: float = field(default_factory=lambda: _env("thermal_ceiling_c", 80.0))
    battery_floor_pct: float = field(default_factory=lambda: _env("battery_floor_pct", 20.0))


@dataclass(slots=True)
class SyncConfig:
    enabled: bool = field(default_factory=lambda: _env("sync_enabled", True))
    cloud_url: str = field(default_factory=lambda: _env("cloud_url", "loopback://cloud"))
    interval_s: float = field(default_factory=lambda: _env("sync_interval_s", 12.0))
    batch_ops: int = field(default_factory=lambda: _env("sync_batch_ops", 256))
    bandwidth_bps: int = field(default_factory=lambda: _env("sync_bandwidth_bps", 2_000_000))
    probe_interval_s: float = field(default_factory=lambda: _env("probe_interval_s", 3.0))
    merkle_fanout: int = field(default_factory=lambda: _env("merkle_fanout", 16))


@dataclass(slots=True)
class RenewalConfig:
    enabled: bool = field(default_factory=lambda: _env("renewal_enabled", True))
    interval_s: float = field(default_factory=lambda: _env("renewal_interval_s", 30.0))
    ttl_days: float = field(default_factory=lambda: _env("ttl_days", 90.0))
    half_life_days: float = field(default_factory=lambda: _env("half_life_days", 21.0))
    batch: int = field(default_factory=lambda: _env("renewal_batch", 256))


@dataclass(slots=True)
class LearningConfig:
    enabled: bool = field(default_factory=lambda: _env("learning_enabled", True))
    rank: int = field(default_factory=lambda: _env("adapter_rank", 16))
    alpha: float = field(default_factory=lambda: _env("adapter_alpha", 0.35))
    batch: int = field(default_factory=lambda: _env("learning_batch", 4))
    differential_privacy: bool = field(default_factory=lambda: _env("differential_privacy", True))
    epsilon_total: float = field(default_factory=lambda: _env("epsilon_total", 8.0))
    epsilon_per_round: float = field(default_factory=lambda: _env("epsilon_per_round", 2.0))


@dataclass(slots=True)
class Settings:
    node_id: str = field(default_factory=lambda: _env("node_id", "edge-07"))
    data_dir: Path = field(default_factory=lambda: Path(_env("data_dir", ".aegis")))
    host: str = field(default_factory=lambda: _env("host", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env("port", 8000))
    cors_origins: str = field(default_factory=lambda: _env("cors_origins", "*"))
    policy_file: str = field(default_factory=lambda: _env("policy_file", "config/policy.yaml"))
    seed_demo: bool = field(default_factory=lambda: _env("seed_demo", True))
    telemetry_interval_s: float = field(default_factory=lambda: _env("telemetry_interval_s", 2.0))
    scheduler_concurrency: int = field(default_factory=lambda: _env("scheduler_concurrency", 3))
    mesh_enabled: bool = field(default_factory=lambda: _env("mesh_enabled", True))
    mesh_interval_s: float = field(default_factory=lambda: _env("mesh_interval_s", 15.0))

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    renewal: RenewalConfig = field(default_factory=RenewalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["memory"]["collections"] = list(self.memory.collections)
        return d


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
    return _settings
