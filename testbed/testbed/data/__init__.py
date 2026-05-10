"""Data helpers with lazy optional-dependency imports.

Importing ``testbed.data.schema`` should not require h5py or torch.  The heavy
HDF5/dataset helpers are loaded only when their public names are requested.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "write_episode",
    "read_episode",
    "list_episodes",
    "EpisodeRecorder",
    "EpisodicDataset",
    "get_norm_stats",
    "load_data",
]


def __getattr__(name: str) -> Any:
    if name in {"write_episode", "read_episode", "list_episodes"}:
        from testbed.data.hdf5_io import list_episodes, read_episode, write_episode

        values = {
            "write_episode": write_episode,
            "read_episode": read_episode,
            "list_episodes": list_episodes,
        }
        return values[name]

    if name == "EpisodeRecorder":
        from testbed.data.recorder import EpisodeRecorder

        return EpisodeRecorder

    if name in {"EpisodicDataset", "get_norm_stats", "load_data"}:
        from testbed.data.dataset import EpisodicDataset, get_norm_stats, load_data

        values = {
            "EpisodicDataset": EpisodicDataset,
            "get_norm_stats": get_norm_stats,
            "load_data": load_data,
        }
        return values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
