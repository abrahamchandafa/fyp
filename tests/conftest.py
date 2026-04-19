"""Pytest configuration for test discovery and imports."""

import importlib.util
import sys
from pathlib import Path


def load_module(module_name, file_path):
    """Load a Python module from a file path, avoiding sys.path collision."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Helpers for loading specific modules
def load_generation_module():
    """Load generation/generate.py."""
    gen_path = Path(__file__).parent.parent / "generation" / "generate.py"
    return load_module("gen_module", str(gen_path))


def load_track_common():
    """Load track/common.py with unique module name."""
    track_path = Path(__file__).parent.parent / "track" / "common.py"
    return load_module("track_module_common", str(track_path))


def load_pose_common():
    """Load pose/common.py with unique module name."""
    pose_path = Path(__file__).parent.parent / "pose" / "common.py"
    return load_module("pose_module_common", str(pose_path))


def load_pdr_model():
    """Load pdr/model.py."""
    pdr_path = Path(__file__).parent.parent / "pdr" / "model.py"
    return load_module("pdr_module", str(pdr_path))
