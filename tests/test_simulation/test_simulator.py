"""Tests for simulator."""

import pytest
import numpy as np
from src.simulation.simulator import (
    SimulationConfig,
    SimulationLevel,
    SimulationResult,
    TimeStep,
    EPSSimulator,
    PopulationGenerator,
    SignalGenerator,
)


class TestSimulationLevel:
    """Tests for SimulationLevel enum."""

    def test_level_values(self):
        assert SimulationLevel.LEVEL1_AGENT.value == 1


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        config = SimulationConfig()
        assert config.num_devices == 5000
        assert config.time_step == 60.0
        assert config.level == SimulationLevel.LEVEL1_AGENT

    def test_custom_config(self):
        config = SimulationConfig(
            num_devices=5000,
            duration_seconds=7200,
            random_seed=123,
        )
        assert config.num_devices == 5000
        assert config.duration_seconds == 7200

    def test_battery_only_fraction(self):
        config = SimulationConfig()
        assert config.battery_fraction == 1.0


class TestTimeStep:
    """Tests for TimeStep."""

    def test_default_result(self):
        from datetime import datetime
        result = TimeStep(step_index=0, timestamp=datetime.now(), elapsed_seconds=0.0)
        assert result.total_response_kw == 0.0
        assert result.responding_devices == 0

    def test_with_data(self):
        from datetime import datetime
        result = TimeStep(
            step_index=1,
            timestamp=datetime.now(),
            elapsed_seconds=60.0,
            total_response_kw=1000.0,
            responding_devices=50,
            latency_samples=[50, 60, 70],
        )
        assert result.total_response_kw == 1000.0
        assert len(result.latency_samples) == 3


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_empty_result(self):
        result = SimulationResult()
        assert result.response_rate == 0.0
        assert result.total_energy_kwh == 0.0

    def test_with_time_steps(self):
        from datetime import datetime
        ts1 = TimeStep(step_index=0, timestamp=datetime.now(), elapsed_seconds=0.0, total_response_kw=100.0)
        ts2 = TimeStep(step_index=1, timestamp=datetime.now(), elapsed_seconds=60.0, total_response_kw=200.0)

        result = SimulationResult(time_steps=[ts1, ts2])
        assert len(result.time_steps) == 2

    def test_latency_percentiles(self):
        from datetime import datetime
        ts = TimeStep(
            step_index=0,
            timestamp=datetime.now(),
            elapsed_seconds=0.0,
            latency_samples=list(range(1, 101))
        )
        result = SimulationResult(time_steps=[ts])
        result.compute_latency_percentiles()

        assert 'p50' in result.latency_percentiles
        assert 'p99' in result.latency_percentiles


class TestPopulationGenerator:
    """Tests for PopulationGenerator."""

    def test_generate_population(self):
        config = SimulationConfig(num_devices=100, random_seed=42)
        rng = np.random.default_rng(42)
        generator = PopulationGenerator(config, rng)
        population = generator.generate()

        assert len(population.batteries) == 100

    def test_all_devices_are_batteries(self):
        config = SimulationConfig(
            num_devices=1000,
            random_seed=42,
        )
        rng = np.random.default_rng(42)
        generator = PopulationGenerator(config, rng)
        population = generator.generate()

        assert len(population.batteries) == 1000


class TestSignalGenerator:
    """Tests for SignalGenerator."""

    def test_generate_signal(self):
        config = SimulationConfig(random_seed=42)
        rng = np.random.default_rng(42)
        generator = SignalGenerator(config, rng)

        signal = generator.generate_signal(
            region_id=0,
            supply_demand=7,
            intensity=2000,
        )

        assert signal.region_id == 0
        assert 0 <= signal.intensity <= 4095
        assert 0 <= signal.supply_demand <= 15

    def test_generate_scenario_signals(self):
        config = SimulationConfig(num_regions=4, random_seed=42)
        rng = np.random.default_rng(42)
        generator = SignalGenerator(config, rng)

        signals = generator.generate_scenario_signals('peak_shaving', time_step=0, num_regions=4)

        assert len(signals) == 4


class TestEPSSimulator:
    """Tests for EPSSimulator."""

    @pytest.fixture
    def small_simulator(self):
        config = SimulationConfig(
            num_devices=100,
            duration_seconds=300,
            time_step=60.0,
            random_seed=42,
        )
        return EPSSimulator(config)

    def test_initialization(self, small_simulator):
        assert small_simulator.config.num_devices == 100

    def test_run_single_step(self, small_simulator):
        result = small_simulator.run(num_steps=1, scenario='peak_shaving')

        assert len(result.time_steps) == 1
        assert result.time_steps[0].step_index == 0

    def test_run_multiple_steps(self, small_simulator):
        result = small_simulator.run(num_steps=5, scenario='peak_shaving')

        assert len(result.time_steps) == 5
        for i, ts in enumerate(result.time_steps):
            assert ts.step_index == i

    def test_response_rate_in_range(self, small_simulator):
        result = small_simulator.run(num_steps=5, scenario='peak_shaving')

        assert result.response_rate >= 0.0
        assert result.response_rate <= 1.0

    def test_different_scenarios(self, small_simulator):
        scenarios = ['peak_shaving', 'valley_filling', 'emergency', 'normal']

        for scenario in scenarios:
            result = small_simulator.run(num_steps=2, scenario=scenario)
            assert len(result.time_steps) == 2

    def test_energy_tracking(self, small_simulator):
        result = small_simulator.run(num_steps=5, scenario='peak_shaving')

        # Energy can be positive (discharge) or negative (charge) depending on scenario
        assert np.isfinite(result.total_energy_kwh)

    def test_device_breakdown(self, small_simulator):
        result = small_simulator.run(num_steps=3, scenario='peak_shaving')

        # Battery-only: check battery response is finite
        assert np.isfinite(result.battery_response_kwh)

    def test_reproducibility(self):
        config = SimulationConfig(num_devices=50, random_seed=42)

        sim1 = EPSSimulator(config)
        result1 = sim1.run(num_steps=3, scenario='peak_shaving')

        sim2 = EPSSimulator(config)
        result2 = sim2.run(num_steps=3, scenario='peak_shaving')

        assert result1.response_rate == result2.response_rate

