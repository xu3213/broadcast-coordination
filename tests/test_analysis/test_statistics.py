"""Tests for statistical analysis module."""

import pytest
import numpy as np
from src.analysis.statistics import (
    BootstrapResult,
    EffectSizeResult,
    HypothesisTestResult,
    NormalityTestResult,
    ComparisonSummary,
    StatisticalAnalyzer,
    generate_statistical_report,
)


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_default_result(self):
        result = BootstrapResult(
            point_estimate=10.0,
            ci_lower=8.0,
            ci_upper=12.0,
        )
        assert result.confidence_level == 0.95
        assert result.n_bootstrap == 10000

    def test_str_representation(self):
        result = BootstrapResult(
            point_estimate=10.0,
            ci_lower=8.0,
            ci_upper=12.0,
        )
        s = str(result)
        assert "10.0000" in s
        assert "95%" in s


class TestEffectSizeResult:
    """Tests for EffectSizeResult dataclass."""

    def test_result(self):
        result = EffectSizeResult(cohens_d=0.8, interpretation='large')
        assert result.cohens_d == 0.8
        assert result.interpretation == 'large'

    def test_str_representation(self):
        result = EffectSizeResult(cohens_d=0.5, interpretation='medium')
        s = str(result)
        assert "0.500" in s
        assert "medium" in s


class TestHypothesisTestResult:
    """Tests for HypothesisTestResult dataclass."""

    def test_significant_result(self):
        result = HypothesisTestResult(
            statistic=2.5,
            p_value=0.01,
            test_name="t-test",
            significant=True,
        )
        assert result.significant is True

    def test_str_representation(self):
        result = HypothesisTestResult(
            statistic=2.5,
            p_value=0.01,
            test_name="t-test",
            significant=True,
        )
        s = str(result)
        assert "t-test" in s
        assert "significant" in s


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        return StatisticalAnalyzer(random_seed=42)

    @pytest.fixture
    def normal_data(self):
        np.random.seed(42)
        return np.random.normal(10, 2, 100)

    @pytest.fixture
    def two_groups(self):
        np.random.seed(42)
        group1 = np.random.normal(12, 2, 50)
        group2 = np.random.normal(10, 2, 50)
        return group1, group2

    def test_bootstrap_ci_percentile(self, analyzer, normal_data):
        result = analyzer.bootstrap_ci(
            normal_data,
            n_bootstrap=1000,
            method='percentile',
        )

        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.method == 'percentile'

    def test_bootstrap_ci_bca(self, analyzer, normal_data):
        result = analyzer.bootstrap_ci(
            normal_data,
            n_bootstrap=1000,
            method='BCa',
        )

        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.method == 'BCa'

    def test_bootstrap_ci_empty(self, analyzer):
        result = analyzer.bootstrap_ci(np.array([]))
        assert result.point_estimate == 0.0

    def test_cohens_d_large_effect(self, analyzer, two_groups):
        group1, group2 = two_groups
        result = analyzer.cohens_d(group1, group2)

        assert result.cohens_d > 0.5  # Significant effect
        assert result.interpretation in ['medium', 'large']

    def test_cohens_d_no_difference(self, analyzer):
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        group1 = data[:50]
        group2 = data[:50] + np.random.normal(0, 0.1, 50)  # Small noise

        result = analyzer.cohens_d(group1, group2)
        assert abs(result.cohens_d) < 0.5  # Small effect

    def test_cohens_d_empty(self, analyzer):
        result = analyzer.cohens_d(np.array([]), np.array([1, 2, 3]))
        assert result.interpretation == 'undefined'

    def test_test_normality_normal_data(self, analyzer, normal_data):
        result = analyzer.test_normality(normal_data)
        assert bool(result.is_normal) is True
        assert result.test_name == "Shapiro-Wilk"

    def test_test_normality_non_normal(self, analyzer):
        # Highly skewed data
        np.random.seed(42)
        data = np.random.exponential(2, 100)
        result = analyzer.test_normality(data)
        assert bool(result.is_normal) is False

    def test_mann_whitney_u(self, analyzer, two_groups):
        group1, group2 = two_groups
        result = analyzer.mann_whitney_u(group1, group2)

        assert result.test_name == "Mann-Whitney U"
        assert result.p_value < 0.05  # Should be significant

    def test_welch_t_test(self, analyzer, two_groups):
        group1, group2 = two_groups
        result = analyzer.welch_t_test(group1, group2)

        assert result.test_name == "Welch's t-test"
        assert result.p_value < 0.05

    def test_benjamini_hochberg(self, analyzer):
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5]
        significant, adjusted = analyzer.benjamini_hochberg(p_values)

        assert len(significant) == 6
        assert len(adjusted) == 6
        assert significant[0] is True  # Most significant
        assert adjusted[0] <= adjusted[-1]  # Adjusted p-values should be ordered

    def test_benjamini_hochberg_empty(self, analyzer):
        significant, adjusted = analyzer.benjamini_hochberg([])
        assert significant == []
        assert adjusted == []

    def test_compute_power(self, analyzer):
        power = analyzer.compute_power(effect_size=0.8, n=30)
        assert 0.7 < power < 1.0  # Should have good power

    def test_compute_power_small_effect(self, analyzer):
        power = analyzer.compute_power(effect_size=0.2, n=30)
        assert power < 0.5  # Low power for small effect

    def test_required_sample_size(self, analyzer):
        n = analyzer.required_sample_size(effect_size=0.8, power=0.8)
        assert 20 < n < 50  # Reasonable sample size for large effect

    def test_required_sample_size_small_effect(self, analyzer):
        n = analyzer.required_sample_size(effect_size=0.2, power=0.8)
        assert n > 200  # Large sample needed for small effect

    def test_summarize_comparison(self, analyzer, two_groups):
        group1, group2 = two_groups
        summary = analyzer.summarize_comparison(
            "Method A", "Method B", "Latency",
            group1, group2,
        )

        assert summary.method1 == "Method A"
        assert summary.method2 == "Method B"
        assert summary.metric == "Latency"
        assert summary.effect_size is not None
        assert summary.test is not None


class TestGenerateStatisticalReport:
    """Tests for generate_statistical_report function."""

    def test_generate_report(self):
        results = {
            'EPS': {
                'latency': [50.0, 55.0, 48.0, 52.0, 51.0] * 6,
                'response_rate': [0.6, 0.62, 0.58, 0.61, 0.59] * 6,
            },
            'OpenADR': {
                'latency': [3000.0, 3100.0, 2900.0, 3050.0, 2950.0] * 6,
                'response_rate': [0.4, 0.42, 0.38, 0.41, 0.39] * 6,
            },
        }

        report = generate_statistical_report(results)

        assert "STATISTICAL ANALYSIS REPORT" in report
        assert "SUMMARY STATISTICS" in report
        assert "PAIRWISE COMPARISONS" in report
        assert "POWER ANALYSIS" in report

    def test_generate_report_single_method(self):
        results = {
            'EPS': {
                'latency': [50.0, 55.0, 48.0] * 10,
            },
        }

        report = generate_statistical_report(results)
        assert "SUMMARY STATISTICS" in report
