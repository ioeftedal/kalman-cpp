"""Shared utilities for kalman-cpp plotting scripts."""

import csv
from pathlib import Path

import numpy as np


def read_csv(path: str) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames or []:
            data[col] = []
        for row in reader:
            for col in reader.fieldnames or []:
                data[col].append(float(row[col]))
    return {k: np.array(v) for k, v in data.items()}
