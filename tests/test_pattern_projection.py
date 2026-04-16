"""Tests for pattern projection."""

import numpy as np
import pytest

from thermal_tracker.pattern_projection import PatternProjector, ThermalDot


class TestThermalDot:
    def test_dot_creation(self) -> None:
        dot = ThermalDot(x=100.0, y=200.0, intensity=15.0, frame_created=0)
        assert dot.x == 100.0
        assert dot.y == 200.0
        assert dot.intensity == 15.0

    def test_dot_is_frozen(self) -> None:
        dot = ThermalDot(x=1.0, y=2.0)
        with pytest.raises(AttributeError):
            dot.x = 5.0  # type: ignore[misc]


class TestPatternProjectorInit:
    def test_default_construction(self) -> None:
        proj = PatternProjector(640, 480)
        assert proj.w == 640
        assert proj.h == 480

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="margin"):
            PatternProjector(30, 30, margin=20)  # 30 <= 2*20


class TestGridPattern:
    def test_grid_count(self) -> None:
        proj = PatternProjector(640, 480, margin=20)
        dots = proj.generate_grid(rows=3, cols=4)
        assert len(dots) == 12

    def test_grid_within_margins(self) -> None:
        proj = PatternProjector(640, 480, margin=30)
        dots = proj.generate_grid(rows=5, cols=5)
        for d in dots:
            assert d.x >= 30
            assert d.x <= 610
            assert d.y >= 30
            assert d.y <= 450

    def test_grid_respects_max_dots(self) -> None:
        proj = PatternProjector(640, 480, max_dots=5)
        dots = proj.generate_grid(rows=4, cols=4)
        assert len(dots) <= 5


class TestRandomPattern:
    def test_random_count(self) -> None:
        proj = PatternProjector(640, 480, rng_seed=42)
        dots = proj.generate_random(n=10)
        assert len(dots) == 10

    def test_random_within_margins(self) -> None:
        proj = PatternProjector(640, 480, margin=50, rng_seed=42)
        dots = proj.generate_random(n=8)
        for d in dots:
            assert d.x >= 50
            assert d.x <= 590
            assert d.y >= 50
            assert d.y <= 430

    def test_random_min_separation(self) -> None:
        proj = PatternProjector(640, 480, min_separation=40.0, rng_seed=42)
        dots = proj.generate_random(n=8)
        for i, a in enumerate(dots):
            for b in dots[i + 1 :]:
                dist = np.hypot(a.x - b.x, a.y - b.y)
                assert dist >= 40.0

    def test_reproducibility(self) -> None:
        proj1 = PatternProjector(640, 480, rng_seed=99)
        dots1 = proj1.generate_random(n=6)
        proj2 = PatternProjector(640, 480, rng_seed=99)
        dots2 = proj2.generate_random(n=6)
        for a, b in zip(dots1, dots2):
            assert a.x == b.x
            assert a.y == b.y


class TestAdaptivePattern:
    def test_adaptive_replaces_faded_dots(self) -> None:
        proj = PatternProjector(640, 480, rng_seed=42)
        proj.generate_random(n=6, frame_index=0)

        # Simulate a field where all dots have faded
        faded_field = np.zeros((480, 640))
        dots = proj.generate_adaptive(
            faded_field, threshold=1.0, n_new=4, frame_index=1
        )
        # Old dots should be gone, new ones should appear
        assert len(dots) <= proj.max_dots
        assert len(dots) >= 4  # At least the new ones

    def test_adaptive_keeps_strong_dots(self) -> None:
        proj = PatternProjector(640, 480, rng_seed=42)
        initial = proj.generate_random(n=6, frame_index=0)

        # Field where all dot positions are still hot
        hot_field = np.zeros((480, 640))
        for d in initial:
            ix, iy = int(round(d.x)), int(round(d.y))
            hot_field[iy, ix] = 10.0

        dots = proj.generate_adaptive(hot_field, threshold=1.0, n_new=2, frame_index=1)
        assert len(dots) >= 6  # originals survived + new ones


class TestActiveDotsAccessor:
    def test_active_dots_returns_copy(self) -> None:
        proj = PatternProjector(640, 480, rng_seed=42)
        proj.generate_random(n=5)
        dots = proj.active_dots
        dots.clear()
        assert len(proj.active_dots) == 5  # original unchanged
