"""Tests for generation module."""

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Load generation/generate.py by file path
gen_path = Path(__file__).parent.parent / "generation" / "generate.py"
spec = importlib.util.spec_from_file_location("gen_module", str(gen_path))
generate = importlib.util.module_from_spec(spec)
sys.modules["gen_module"] = generate
spec.loader.exec_module(generate)

Dot = generate.Dot
shift_translate_x = generate.shift_translate_x
shift_translate_y = generate.shift_translate_y
shift_translate_z = generate.shift_translate_z
shift_rotate_yaw = generate.shift_rotate_yaw
shift_rotate_pitch = generate.shift_rotate_pitch
shift_rotate_roll = generate.shift_rotate_roll
DOF_REGISTRY = generate.DOF_REGISTRY
IMAGE_WIDTH = generate.IMAGE_WIDTH
IMAGE_HEIGHT = generate.IMAGE_HEIGHT
SHIFT_RATE = generate.SHIFT_RATE
ZOOM_RATE = generate.ZOOM_RATE
ROTATION_RATE = generate.ROTATION_RATE


class TestDot:
    """Test Dot dataclass."""

    def test_dot_creation(self):
        """Test creating a Dot instance."""
        dot = Dot(x=10.0, y=20.0, intensity=150.0, radius=5.0)
        assert dot.x == 10.0
        assert dot.y == 20.0
        assert dot.intensity == 150.0
        assert dot.radius == 5.0

    def test_dot_modification(self):
        """Test modifying Dot attributes."""
        dot = Dot(x=10.0, y=20.0, intensity=150.0, radius=5.0)
        dot.x = 15.0
        dot.intensity = 100.0
        assert dot.x == 15.0
        assert dot.intensity == 100.0


class TestShiftTranslateX:
    """Test translate_x shift function."""

    def test_translate_x_shifts_left(self):
        """Dots move left with translate_x."""
        dot = Dot(x=100.0, y=50.0, intensity=100.0, radius=5.0)
        dots = [dot]
        shift_translate_x(dots)
        assert dot.x == 100.0 - SHIFT_RATE
        assert dot.y == 50.0  # y unchanged

    def test_translate_x_multiple_dots(self):
        """translate_x shifts all dots correctly."""
        dots = [
            Dot(x=100.0, y=50.0, intensity=100.0, radius=5.0),
            Dot(x=200.0, y=150.0, intensity=120.0, radius=6.0),
        ]
        shift_translate_x(dots)
        assert dots[0].x == 100.0 - SHIFT_RATE
        assert dots[1].x == 200.0 - SHIFT_RATE
        assert dots[0].y == 50.0
        assert dots[1].y == 150.0


class TestShiftTranslateY:
    """Test translate_y shift function."""

    def test_translate_y_shifts_down(self):
        """Dots move down with translate_y."""
        dot = Dot(x=100.0, y=50.0, intensity=100.0, radius=5.0)
        dots = [dot]
        shift_translate_y(dots)
        assert dot.y == 50.0 + SHIFT_RATE
        assert dot.x == 100.0  # x unchanged

    def test_translate_y_multiple_dots(self):
        """translate_y shifts all dots correctly."""
        dots = [
            Dot(x=100.0, y=50.0, intensity=100.0, radius=5.0),
            Dot(x=200.0, y=150.0, intensity=120.0, radius=6.0),
        ]
        shift_translate_y(dots)
        assert dots[0].y == 50.0 + SHIFT_RATE
        assert dots[1].y == 150.0 + SHIFT_RATE
        assert dots[0].x == 100.0
        assert dots[1].x == 200.0


class TestShiftTranslateZ:
    """Test translate_z shift function (zoom)."""

    def test_translate_z_expands_from_center(self):
        """Dots expand radially from center."""
        cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
        dot = Dot(x=cx + 100.0, y=cy, intensity=100.0, radius=5.0)
        dots = [dot]
        shift_translate_z(dots)
        assert dot.x > cx + 100.0  # moved further from center
        assert dot.y == cy  # no vertical change from center

    def test_translate_z_center_stays(self):
        """Dot at center doesn't move."""
        cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
        dot = Dot(x=cx, y=cy, intensity=100.0, radius=5.0)
        dots = [dot]
        original_x, original_y = dot.x, dot.y
        shift_translate_z(dots)
        assert dot.x == original_x
        assert dot.y == original_y


class TestShiftRotateYaw:
    """Test rotate_yaw shift function."""

    def test_rotate_yaw_shifts_left(self):
        """Dots shift left during yaw rotation."""
        cx = IMAGE_WIDTH / 2
        dot = Dot(x=cx, y=100.0, intensity=100.0, radius=5.0)
        dots = [dot]
        shift_rotate_yaw(dots)
        assert dot.x < cx

    def test_rotate_yaw_center_moves_more(self):
        """Dots further from center move more in yaw."""
        cx = IMAGE_WIDTH / 2
        dot_center = Dot(x=cx, y=100.0, intensity=100.0, radius=5.0)
        dot_edge = Dot(x=cx + 200.0, y=100.0, intensity=100.0, radius=5.0)
        dots_center = [dot_center]
        dots_edge = [dot_edge]
        shift_rotate_yaw(dots_center)
        shift_rotate_yaw(dots_edge)
        assert abs(dot_edge.x - (cx + 200.0)) > abs(dot_center.x - cx)


class TestShiftRotatePitch:
    """Test rotate_pitch shift function."""

    def test_rotate_pitch_shifts_down(self):
        """Dots shift down during pitch rotation."""
        cy = IMAGE_HEIGHT / 2
        dot = Dot(x=100.0, y=cy, intensity=100.0, radius=5.0)
        dots = [dot]
        shift_rotate_pitch(dots)
        assert dot.y > cy

    def test_rotate_pitch_edge_moves_more(self):
        """Dots further from center move more in pitch."""
        cy = IMAGE_HEIGHT / 2
        dot_center = Dot(x=100.0, y=cy, intensity=100.0, radius=5.0)
        dot_edge = Dot(x=100.0, y=cy + 200.0, intensity=100.0, radius=5.0)
        dots_center = [dot_center]
        dots_edge = [dot_edge]
        shift_rotate_pitch(dots_center)
        shift_rotate_pitch(dots_edge)
        assert abs(dot_edge.y - (cy + 200.0)) > abs(dot_center.y - cy)


class TestShiftRotateRoll:
    """Test rotate_roll shift function."""

    def test_rotate_roll_rotates_around_center(self):
        """Dots rotate around image center."""
        cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
        # Dot on +x axis
        dot = Dot(x=cx + 100.0, y=cy, intensity=100.0, radius=5.0)
        dots = [dot]
        original_dist = math.sqrt((dot.x - cx) ** 2 + (dot.y - cy) ** 2)
        shift_rotate_roll(dots)
        new_dist = math.sqrt((dot.x - cx) ** 2 + (dot.y - cy) ** 2)
        # Distance should be approximately preserved
        assert abs(new_dist - original_dist) < 0.1

    def test_rotate_roll_changes_angle(self):
        """Dot angle changes after roll rotation."""
        cx, cy = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
        dot = Dot(x=cx + 100.0, y=cy, intensity=100.0, radius=5.0)
        dots = [dot]
        original_angle = math.atan2(dot.y - cy, dot.x - cx)
        shift_rotate_roll(dots)
        new_angle = math.atan2(dot.y - cy, dot.x - cx)
        assert original_angle != new_angle


class TestDofRegistry:
    """Test DoF registry."""

    def test_dof_registry_contains_all_six_dofs(self):
        """Registry contains all 6 DoF shift functions."""
        expected_dofs = [
            "translate_x",
            "translate_y",
            "translate_z",
            "rotate_yaw",
            "rotate_pitch",
            "rotate_roll",
        ]
        for dof in expected_dofs:
            assert dof in DOF_REGISTRY

    def test_dof_registry_functions_are_callable(self):
        """All registry entries are callable."""
        for dof, func in DOF_REGISTRY.items():
            assert callable(func), f"{dof} is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
