"""Reader for the GDF 1.25 recordings in the Dreyer2023 database.

The files are written by OpenViBE and use a fixed 256-byte header followed by
one 256-byte block per channel. Records hold a single sample per channel, so
the data section is a plain (n_records, n_channels) array.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# OpenViBE / GDF stimulation codes observed in this database.
EVENT_CODES = {
    768: "trial_start",
    769: "cue_left",
    770: "cue_right",
    781: "feedback",
    786: "cross",
    800: "trial_end",
    1010: "label_misc",
    32769: "experiment_start",
    32770: "experiment_stop",
    32775: "baseline_start",
    32776: "baseline_stop",
    33281: "segment_start",
    33282: "beep",
}

CUE_CODES = {769: "left", 770: "right"}

# The 27 EEG channels in acquisition order with EOG/EMG removed. This is the
# column order of the saved online CSP matrices, because the online scenario
# used an exclusion selector ("!EOG1;EOG2;EOG3;EMGg;EMGd") that preserves the
# order of the remaining channels.
EEG_CHANNELS = [
    "Fz", "FCz", "Cz", "CPz", "Pz",
    "C1", "C3", "C5", "C2", "C4", "C6",
    "F4", "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4",
    "F3", "FC1", "FC3", "FC5", "CP1", "CP3", "CP5", "P3",
]
EOG_CHANNELS = ["EOG1", "EOG2", "EOG3"]
EMG_CHANNELS = ["EMGg", "EMGd"]

# Physical channel order as stored in every GDF file (verified identical
# across all 694 recordings): EOG and EMG are interleaved at indices 11-15.
FILE_CHANNEL_ORDER = (
    EEG_CHANNELS[:11] + EOG_CHANNELS + EMG_CHANNELS + EEG_CHANNELS[11:]
)


@dataclass
class GdfHeader:
    path: Path
    version: str
    n_channels: int
    n_records: int
    sfreq: float
    header_bytes: int
    bytes_per_sample: int
    ch_names: list[str]
    n_events: int

    @property
    def n_samples(self) -> int:
        return self.n_records

    @property
    def duration_s(self) -> float:
        return self.n_records / self.sfreq


def _infer_bytes_per_sample(
    file_size: int, header_bytes: int, n_records: int, n_channels: int
) -> tuple[int, int]:
    """Recover sample width from file geometry.

    The event table is 8 bytes of header plus 6 bytes per event in mode 1, so
    only one sample width leaves a consistent remainder.
    """
    for bps in (8, 4, 2, 1):
        data_bytes = n_records * n_channels * bps
        tail = file_size - header_bytes - data_bytes
        if tail < 0:
            continue
        if tail == 0:
            return bps, 0
        if tail >= 8 and (tail - 8) % 6 == 0:
            return bps, (tail - 8) // 6
    raise ValueError(f"cannot infer sample width for {file_size} bytes")


def read_header(path: str | Path) -> GdfHeader:
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fixed = fh.read(256)
        version = fixed[:8].decode("latin1").strip()
        header_bytes = struct.unpack("<H", fixed[184:186])[0]
        n_records = struct.unpack("<q", fixed[236:244])[0]
        dur_num, dur_den = struct.unpack("<II", fixed[244:252])
        n_channels = struct.unpack("<H", fixed[252:254])[0]

        # Some writers store the header length in 256-byte blocks instead.
        if header_bytes == n_channels + 1:
            header_bytes *= 256
        if header_bytes != 256 * (n_channels + 1):
            raise ValueError(
                f"{path.name}: header {header_bytes} inconsistent with "
                f"{n_channels} channels"
            )

        var = fh.read(header_bytes - 256)
        ch_names = [
            var[i * 16 : (i + 1) * 16].decode("latin1").strip()
            for i in range(n_channels)
        ]

    if dur_num == 0:
        raise ValueError(f"{path.name}: zero record duration")
    sfreq = dur_den / dur_num
    bps, n_events = _infer_bytes_per_sample(size, header_bytes, n_records, n_channels)

    return GdfHeader(
        path=path,
        version=version,
        n_channels=n_channels,
        n_records=n_records,
        sfreq=sfreq,
        header_bytes=header_bytes,
        bytes_per_sample=bps,
        ch_names=ch_names,
        n_events=n_events,
    )


def read_events(path: str | Path, header: GdfHeader | None = None) -> np.ndarray:
    """Return an (n_events, 2) array of [sample_index, event_code].

    GDF stores 1-based sample positions; they are converted to 0-based here.
    """
    header = header or read_header(path)
    if header.n_events == 0:
        return np.zeros((0, 2), dtype=np.int64)

    offset = (
        header.header_bytes
        + header.n_records * header.n_channels * header.bytes_per_sample
    )
    with open(path, "rb") as fh:
        fh.seek(offset)
        head = fh.read(8)
        n_events = struct.unpack("<I", head[4:8])[0]
        if n_events != header.n_events:
            raise ValueError(
                f"{Path(path).name}: event count {n_events} != inferred "
                f"{header.n_events}"
            )
        pos = np.frombuffer(fh.read(4 * n_events), dtype="<u4").astype(np.int64)
        typ = np.frombuffer(fh.read(2 * n_events), dtype="<u2").astype(np.int64)

    return np.column_stack([np.maximum(pos - 1, 0), typ])


def read_gdf(
    path: str | Path,
    picks: list[str] | None = None,
    dtype: type = np.float32,
) -> tuple[np.ndarray, GdfHeader, np.ndarray]:
    """Return (data[n_channels, n_samples], header, events).

    Data are memory-mapped and copied only for the requested channels, which
    keeps peak memory near the size of the picked subset rather than the file.
    """
    header = read_header(path)
    if header.bytes_per_sample != 8:
        raise NotImplementedError(
            f"{Path(path).name}: expected float64 samples, "
            f"got {header.bytes_per_sample} bytes"
        )

    raw = np.memmap(
        path,
        dtype="<f8",
        mode="r",
        offset=header.header_bytes,
        shape=(header.n_records, header.n_channels),
    )

    if picks is None:
        idx = np.arange(header.n_channels)
    else:
        lookup = {name: i for i, name in enumerate(header.ch_names)}
        missing = [p for p in picks if p not in lookup]
        if missing:
            raise KeyError(f"{Path(path).name}: missing channels {missing}")
        idx = np.array([lookup[p] for p in picks])

    data = np.ascontiguousarray(raw[:, idx].T.astype(dtype))
    del raw
    events = read_events(path, header)
    return data, header, events


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
