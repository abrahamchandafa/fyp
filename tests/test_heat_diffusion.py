"""Tests for the HeatDiffusionSimulator (fade-and-erode model)."""

import numpy as np
import pytest

from thermal_tracker.heat_diffusion import HeatDiffusionSimulator


class TestHeatDiffusionInit:
    """Constructor validation."""

    def test_default_construction(self) -> None:
        sim = HeatDiffusionSimulator(100, 80)
        assert sim.width == 100
        assert sim.height == 80
        assert sim.field.shape == (80, 100)
        np.testing.assert_allclose(sim.field, sim.ambient)

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HeatDiffusionSimulator(0, 10)
        with pytest.raises(ValueError, match="positive"):
            HeatDiffusionSimulator(10, -5)

    def test_invalid_fade_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="fade_rate"):
            HeatDiffusionSimulator(10, 10, fade_rate=-1)

    def test_invalid_erode_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="erode_rate"):
            HeatDiffusionSimulator(10, 10, erode_rate=-0.5)

    def test_invalid_min_intensity_raises(self) -> None:
        with pytest.raises(ValueError, match="min_intensity"):
            HeatDiffusionSimulator(10, 10, min_intensity=-1)

    def test_invalid_min_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="min_radius"):
            HeatDiffusionSimulator(10, 10, min_radius=-1)


class TestHeatDeposit:
    """Heat deposition tests."""

    def test_deposit_increases_temperature(self) -> None:
        sim = HeatDiffusionSimulator(100, 100)
        sim.deposit_heat(50, 50, intensity=10.0, radius=3)
        assert sim.field[50, 50] > sim.ambient + 9.0

    def test_deposit_is_localised(self) -> None:
        sim = HeatDiffusionSimulator(200, 200)
        sim.deposit_heat(100, 100, intensity=10.0, radius=2)
        assert sim.field[0, 0] == pytest.approx(sim.ambient, abs=1e-6)

    def test_negative_intensity_raises(self) -> None:
        sim = HeatDiffusionSimulator(50, 50)
        with pytest.raises(ValueError, match="non-negative"):
            sim.deposit_heat(25, 25, intensity=-5)

    def test_deposit_at_edge(self) -> None:
        sim = HeatDiffusionSimulator(100, 100)
        sim.deposit_heat(0, 0, intensity=5.0, radius=2)
        sim.deposit_heat(99, 99, intensity=5.0, radius=2)
        assert sim.field[0, 0] > sim.ambient

    def test_multiple_deposits_accumulate(self) -> None:
        sim = HeatDiffusionSimulator(100, 100)
        sim.deposit_heat(50, 50, intensity=5.0, radius=2)
        t1 = sim.field[50, 50]
        sim.deposit_heat(50, 50, intensity=5.0, radius=2)
        t2 = sim.field[50, 50]
        assert t2 > t1

    def test_dot_count_increases(self) -> None:
        sim = HeatDiffusionSimulator(100, 100)
        assert sim.dot_count == 0
        sim.deposit_heat(50, 50, intensity=10.0)
        assert sim.dot_count == 1
        sim.deposit_heat(70, 70, intensity=10.0)
        assert sim.dot_count == 2


class TestFadeAndErode:
    """Verify that dots fade in intensity and erode in radius over time."""

    def test_intensity_fades_after_step(self) -> None:
        sim = HeatDiffusionSimulator(100, 100, fade_rate=1.0, erode_rate=0.0)
        sim.deposit_heat(50, 50, intensity=20.0, radius=3)
        peak_before = sim.field[50, 50]
        sim.step(1.0)
        peak_after = sim.field[50, 50]
        assert peak_after < peak_before

    def test_radius_erodes_after_step(self) -> None:
        """A smaller radius produces a narrower Gaussian → less spread."""
        sim = HeatDiffusionSimulator(
            200, 200, fade_rate=0.0, erode_rate=1.0, min_intensity=15.0
        )
        sim.deposit_heat(100, 100, intensity=15.0, radius=10)
        # Before erosion: measure intensity at 1σ distance
        val_before = sim.field[100, 110]  # 10 px away = 1σ

        sim.step(1.0)
        val_after = sim.field[100, 110]
        # After erosion, σ shrank → intensity at same distance drops
        assert val_after < val_before

    def test_intensity_approaches_min(self) -> None:
        """After many steps the intensity converges to min_intensity."""
        sim = HeatDiffusionSimulator(100, 100, fade_rate=2.0, min_intensity=2.0)
        sim.deposit_heat(50, 50, intensity=50.0, radius=3)
        sim.step_n(1.0, 50)
        rel = sim.get_relative_field()
        # Peak should be close to min_intensity (the dot's floor)
        assert rel[50, 50] == pytest.approx(2.0, abs=0.5)

    def test_radius_approaches_min(self) -> None:
        """After many steps the radius converges to min_radius."""
        sim = HeatDiffusionSimulator(
            200, 200, erode_rate=2.0, fade_rate=0.0, min_intensity=10.0, min_radius=1.0
        )
        sim.deposit_heat(100, 100, intensity=10.0, radius=15)
        sim.step_n(1.0, 50)
        # With radius ≈ 1 pixel the dot is very tight.
        # Intensity at 5 px away should be negligible.
        far_val = sim.get_relative_field()[100, 105]
        center_val = sim.get_relative_field()[100, 100]
        assert far_val < center_val * 0.01

    def test_no_decay_when_rate_zero(self) -> None:
        sim = HeatDiffusionSimulator(100, 100, fade_rate=0.0, erode_rate=0.0)
        sim.deposit_heat(50, 50, intensity=10.0, radius=3)
        peak_before = sim.field[50, 50]
        sim.step_n(1.0, 100)
        peak_after = sim.field[50, 50]
        assert peak_after == pytest.approx(peak_before, rel=1e-10)

    def test_negative_dt_raises(self) -> None:
        sim = HeatDiffusionSimulator(50, 50)
        with pytest.raises(ValueError, match="non-negative"):
            sim.step(-1.0)

    def test_dots_pruned_after_full_decay(self) -> None:
        """Dots that reach the floor should eventually be pruned."""
        sim = HeatDiffusionSimulator(100, 100, fade_rate=5.0, min_intensity=0.0)
        sim.deposit_heat(50, 50, intensity=10.0, radius=3)
        assert sim.dot_count == 1
        sim.step_n(1.0, 100)
        assert sim.dot_count == 0

    def test_temperature_stays_above_ambient(self) -> None:
        sim = HeatDiffusionSimulator(50, 50, fade_rate=2.0)
        sim.deposit_heat(25, 25, intensity=5.0)
        sim.step_n(1.0, 100)
        assert np.all(sim.field >= sim.ambient)

    def test_step_n_equivalent_to_repeated_step(self) -> None:
        sim_a = HeatDiffusionSimulator(80, 80, fade_rate=0.5, erode_rate=0.3)
        sim_b = HeatDiffusionSimulator(80, 80, fade_rate=0.5, erode_rate=0.3)
        sim_a.deposit_heat(40, 40, intensity=10.0, radius=3)
        sim_b.deposit_heat(40, 40, intensity=10.0, radius=3)
        sim_a.step_n(0.1, 10)
        for _ in range(10):
            sim_b.step(0.1)
        np.testing.assert_allclose(sim_a.field, sim_b.field, atol=1e-10)


class TestRelativeField:
    def test_relative_field_is_zero_at_ambient(self) -> None:
        sim = HeatDiffusionSimulator(30, 30)
        np.testing.assert_allclose(sim.get_relative_field(), 0.0)

    def test_relative_field_matches_deposit(self) -> None:
        sim = HeatDiffusionSimulator(100, 100)
        sim.deposit_heat(50, 50, intensity=10.0, radius=2)
        rel = sim.get_relative_field()
        assert rel[50, 50] == pytest.approx(10.0, abs=0.5)


class TestReset:
    def test_reset_clears_field(self) -> None:
        sim = HeatDiffusionSimulator(50, 50)
        sim.deposit_heat(25, 25, intensity=10)
        sim.reset()
        np.testing.assert_allclose(sim.field, sim.ambient)

    def test_reset_clears_dots(self) -> None:
        sim = HeatDiffusionSimulator(50, 50)
        sim.deposit_heat(25, 25, intensity=10)
        assert sim.dot_count == 1
        sim.reset()
        assert sim.dot_count == 0
