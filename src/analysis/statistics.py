"""
Statistical Analysis

Bootstrap confidence intervals and hypothesis testing utilities
used for multi-run experiment validation (30-run CI in Result 1).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 10000
    method: str = 'BCa'

    def __str__(self) -> str:
        return (
            f"{self.point_estimate:.4f} "
            f"({self.confidence_level*100:.0f}% CI: "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""
    cohens_d: float
    interpretation: str  # negligible, small, medium, large
    hedge_g: Optional[float] = None  # Bias-corrected for small samples

    def __str__(self) -> str:
        return f"d = {self.cohens_d:.3f} ({self.interpretation})"


@dataclass
class HypothesisTestResult:
    """Result of hypothesis test."""
    statistic: float
    p_value: float
    test_name: str
    significant: bool
    effect_size: Optional[float] = None

    def __str__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return f"{self.test_name}: p = {self.p_value:.4f} ({sig})"


@dataclass
class NormalityTestResult:
    """Result of normality test."""
    statistic: float
    p_value: float
    is_normal: bool
    test_name: str = "Shapiro-Wilk"


@dataclass
class ComparisonSummary:
    """Summary of comparison between methods."""
    method1: str
    method2: str
    metric: str
    mean1: float
    mean2: float
    ci1: BootstrapResult
    ci2: BootstrapResult
    effect_size: EffectSizeResult
    test: HypothesisTestResult


class StatisticalAnalyzer:
    """
    Publication-standard statistical analysis.

    Provides:
    - Bootstrap BCa confidence intervals (bias-corrected accelerated)
    - Cohen's d / Hedge's g effect sizes
    - Multiple comparison corrections (FDR)
    - Normality tests (Shapiro-Wilk)
    - Statistical power analysis

    All methods follow academic publication standards:
    - 95% confidence intervals (BCa bootstrap, 10,000 resamples)
    - Effect sizes must be > 0.5 for meaningful claims
    - 30+ independent runs for statistical significance
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean,
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = 'BCa',
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval using BCa method.

        The BCa (bias-corrected and accelerated) method provides
        second-order accurate confidence intervals.

        Args:
            data: Sample data (1D array)
            statistic: Function to compute statistic (default: mean)
            confidence: Confidence level (0-1)
            n_bootstrap: Number of bootstrap samples
            method: 'BCa' (recommended) or 'percentile'

        Returns:
            BootstrapResult with CI bounds
        """
        data = np.asarray(data)
        n = len(data)

        if n == 0:
            return BootstrapResult(0.0, 0.0, 0.0, confidence, n_bootstrap, method)

        # Point estimate
        theta_hat = statistic(data)

        # Generate bootstrap samples
        bootstrap_indices = self.rng.integers(0, n, size=(n_bootstrap, n))
        bootstrap_samples = data[bootstrap_indices]
        bootstrap_stats = np.array([statistic(s) for s in bootstrap_samples])

        if method == 'percentile':
            # Simple percentile method
            alpha = 1 - confidence
            ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        else:
            # BCa method
            ci_lower, ci_upper = self._bca_interval(
                data, theta_hat, bootstrap_stats, statistic, confidence
            )

        return BootstrapResult(
            point_estimate=float(theta_hat),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=confidence,
            n_bootstrap=n_bootstrap,
            method=method,
        )

    def _bca_interval(
        self,
        data: np.ndarray,
        theta_hat: float,
        bootstrap_stats: np.ndarray,
        statistic: Callable,
        confidence: float,
    ) -> Tuple[float, float]:
        """Compute BCa (bias-corrected accelerated) confidence interval."""
        n = len(data)
        alpha = 1 - confidence

        # Bias correction factor (z0)
        prop_less = np.mean(bootstrap_stats < theta_hat)
        z0 = stats.norm.ppf(prop_less) if 0 < prop_less < 1 else 0.0

        # Acceleration factor (a) via jackknife
        jackknife_stats = np.array([
            statistic(np.delete(data, i)) for i in range(n)
        ])
        jack_mean = np.mean(jackknife_stats)
        jack_diff = jack_mean - jackknife_stats

        numerator = np.sum(jack_diff ** 3)
        denominator = 6 * (np.sum(jack_diff ** 2) ** 1.5)
        a = numerator / denominator if abs(denominator) > 1e-10 else 0.0

        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

        def adjusted_percentile(z_alpha):
            num = z0 + z_alpha
            denom = 1 - a * num
            if abs(denom) < 1e-10:
                return stats.norm.cdf(z0 + z_alpha)
            return stats.norm.cdf(z0 + num / denom)

        pct_lower = adjusted_percentile(z_alpha_lower) * 100
        pct_upper = adjusted_percentile(z_alpha_upper) * 100

        # Clip to valid range
        pct_lower = max(0.5, min(99.5, pct_lower))
        pct_upper = max(0.5, min(99.5, pct_upper))

        ci_lower = np.percentile(bootstrap_stats, pct_lower)
        ci_upper = np.percentile(bootstrap_stats, pct_upper)

        return ci_lower, ci_upper

    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True,
    ) -> EffectSizeResult:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group data
            group2: Second group data
            pooled: Use pooled standard deviation (recommended)

        Returns:
            EffectSizeResult with effect size and interpretation
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        if len(group1) == 0 or len(group2) == 0:
            return EffectSizeResult(0.0, 'undefined')

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        n1, n2 = len(group1), len(group2)

        if n1 < 2 or n2 < 2:
            return EffectSizeResult(0.0, 'undefined')

        if pooled:
            # Pooled standard deviation
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            sd = np.sqrt(pooled_var)
        else:
            # Control group SD
            sd = np.std(group2, ddof=1)

        if sd < 1e-10:
            return EffectSizeResult(0.0, 'undefined')

        d = (mean1 - mean2) / sd

        # Hedge's g correction for small samples
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        g = d * correction

        # Interpretation (Cohen's conventions)
        abs_d = abs(d)
        if abs_d < 0.2:
            interp = 'negligible'
        elif abs_d < 0.5:
            interp = 'small'
        elif abs_d < 0.8:
            interp = 'medium'
        else:
            interp = 'large'

        return EffectSizeResult(cohens_d=d, interpretation=interp, hedge_g=g)

    def test_normality(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
    ) -> NormalityTestResult:
        """
        Test for normality using Shapiro-Wilk test.

        Args:
            data: Sample data
            alpha: Significance level

        Returns:
            NormalityTestResult
        """
        data = np.asarray(data)

        if len(data) < 3:
            return NormalityTestResult(0.0, 1.0, True)

        # Shapiro-Wilk test (best for n < 5000)
        if len(data) <= 5000:
            stat, p = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # D'Agostino-Pearson for larger samples
            stat, p = stats.normaltest(data)
            test_name = "D'Agostino-Pearson"

        return NormalityTestResult(
            statistic=stat,
            p_value=p,
            is_normal=p > alpha,
            test_name=test_name,
        )

    def mann_whitney_u(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
    ) -> HypothesisTestResult:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).

        Args:
            group1: First group data
            group2: Second group data
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTestResult
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        stat, p = stats.mannwhitneyu(group1, group2, alternative=alternative)

        # Rank-biserial correlation as effect size
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * stat) / (n1 * n2)

        return HypothesisTestResult(
            statistic=stat,
            p_value=p,
            test_name="Mann-Whitney U",
            significant=p < alpha,
            effect_size=r,
        )

    def welch_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
    ) -> HypothesisTestResult:
        """
        Welch's t-test (does not assume equal variances).

        Args:
            group1: First group data
            group2: Second group data
            alpha: Significance level
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            HypothesisTestResult
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        stat, p = stats.ttest_ind(group1, group2, equal_var=False,
                                   alternative=alternative)

        effect = self.cohens_d(group1, group2)

        return HypothesisTestResult(
            statistic=stat,
            p_value=p,
            test_name="Welch's t-test",
            significant=p < alpha,
            effect_size=effect.cohens_d,
        )

    def benjamini_hochberg(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Tuple[List[bool], List[float]]:
        """
        Benjamini-Hochberg FDR correction for multiple comparisons.

        Args:
            p_values: List of p-values
            alpha: False discovery rate threshold

        Returns:
            Tuple of (significant decisions, adjusted p-values)
        """
        p_values = np.array(p_values)
        n = len(p_values)

        if n == 0:
            return [], []

        # Sort p-values and get ranks
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        ranks = np.arange(1, n + 1)

        # BH critical values
        bh_critical = ranks * alpha / n

        # Find largest k where p(k) <= k*alpha/n
        significant_mask = sorted_p <= bh_critical
        if not np.any(significant_mask):
            threshold_rank = 0
        else:
            threshold_rank = np.max(np.where(significant_mask)[0]) + 1

        # Adjusted p-values
        adjusted = np.minimum(1, sorted_p * n / ranks)
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Reorder to original order
        reorder = np.argsort(sorted_indices)
        adjusted = adjusted[reorder]
        significant = np.array([i < threshold_rank for i in
                                np.argsort(sorted_indices)])[reorder]

        return significant.tolist(), adjusted.tolist()

    def compute_power(
        self,
        effect_size: float,
        n: int,
        alpha: float = 0.05,
        test_type: str = 'two-sample',
    ) -> float:
        """
        Compute statistical power for given effect size and sample size.

        Args:
            effect_size: Cohen's d
            n: Sample size per group
            alpha: Significance level
            test_type: 'two-sample' or 'one-sample'

        Returns:
            Statistical power (0-1)
        """
        from scipy.stats import nct

        if test_type == 'two-sample':
            df = 2 * n - 2
            ncp = effect_size * np.sqrt(n / 2)
        else:
            df = n - 1
            ncp = effect_size * np.sqrt(n)

        t_crit = stats.t.ppf(1 - alpha / 2, df)
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

        return float(power)

    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """
        Calculate required sample size for given power and effect size.

        Args:
            effect_size: Expected Cohen's d
            power: Desired power (default 0.8)
            alpha: Significance level

        Returns:
            Required sample size per group
        """
        # Binary search for sample size
        n_low, n_high = 5, 10000

        while n_high - n_low > 1:
            n_mid = (n_low + n_high) // 2
            current_power = self.compute_power(effect_size, n_mid, alpha)
            if current_power < power:
                n_low = n_mid
            else:
                n_high = n_mid

        return n_high

    def summarize_comparison(
        self,
        method1_name: str,
        method2_name: str,
        metric_name: str,
        data1: np.ndarray,
        data2: np.ndarray,
    ) -> ComparisonSummary:
        """
        Generate comprehensive comparison summary between two methods.

        Args:
            method1_name: Name of first method
            method2_name: Name of second method
            metric_name: Name of metric being compared
            data1: Data from method 1
            data2: Data from method 2

        Returns:
            ComparisonSummary with all statistics
        """
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        # Bootstrap CIs
        ci1 = self.bootstrap_ci(data1)
        ci2 = self.bootstrap_ci(data2)

        # Effect size
        effect = self.cohens_d(data1, data2)

        # Choose test based on normality
        norm1 = self.test_normality(data1)
        norm2 = self.test_normality(data2)

        if norm1.is_normal and norm2.is_normal:
            test = self.welch_t_test(data1, data2)
        else:
            test = self.mann_whitney_u(data1, data2)

        return ComparisonSummary(
            method1=method1_name,
            method2=method2_name,
            metric=metric_name,
            mean1=float(np.mean(data1)),
            mean2=float(np.mean(data2)),
            ci1=ci1,
            ci2=ci2,
            effect_size=effect,
            test=test,
        )


def generate_statistical_report(
    results: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate comprehensive statistical report for experiment results.

    Args:
        results: Dict mapping method names to metric dicts
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    analyzer = StatisticalAnalyzer(random_seed=42)
    lines = []

    lines.append("=" * 70)
    lines.append("STATISTICAL ANALYSIS REPORT (Publication Standard)")
    lines.append("=" * 70)
    lines.append("")

    # Summary statistics for each method
    lines.append("1. SUMMARY STATISTICS")
    lines.append("-" * 70)

    for method, metrics in results.items():
        lines.append(f"\n{method}:")
        for metric_name, values in metrics.items():
            values = np.array(values)
            ci = analyzer.bootstrap_ci(values)
            lines.append(
                f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f} "
                f"(95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}])"
            )

    # Pairwise comparisons
    if len(results) >= 2:
        lines.append("\n\n2. PAIRWISE COMPARISONS")
        lines.append("-" * 70)

        methods = list(results.keys())
        # Compare EPS against baselines
        if 'EPS' in methods:
            eps_methods = ['EPS']
            other_methods = [m for m in methods if m != 'EPS']
        else:
            eps_methods = [methods[0]]
            other_methods = methods[1:]

        all_p_values = []
        comparisons = []

        for eps in eps_methods:
            for other in other_methods:
                for metric in results[eps].keys():
                    if metric in results[other]:
                        summary = analyzer.summarize_comparison(
                            eps, other, metric,
                            np.array(results[eps][metric]),
                            np.array(results[other][metric]),
                        )
                        comparisons.append(summary)
                        all_p_values.append(summary.test.p_value)

        # FDR correction
        if all_p_values:
            significant, adjusted = analyzer.benjamini_hochberg(all_p_values)

            for i, comp in enumerate(comparisons):
                lines.append(
                    f"\n{comp.method1} vs {comp.method2} ({comp.metric}):"
                )
                lines.append(
                    f"  Mean difference: {comp.mean1 - comp.mean2:.4f}"
                )
                lines.append(f"  Effect size: {comp.effect_size}")
                lines.append(
                    f"  {comp.test.test_name}: p = {comp.test.p_value:.4e} "
                    f"(FDR-adjusted: {adjusted[i]:.4e})"
                )
                sig_str = "SIGNIFICANT" if significant[i] else "not significant"
                lines.append(f"  Conclusion: {sig_str} at FDR = 0.05")

    # Power analysis
    lines.append("\n\n3. POWER ANALYSIS")
    lines.append("-" * 70)

    for d, label in [(0.5, 'medium'), (0.8, 'large')]:
        n_required = analyzer.required_sample_size(d, power=0.8)
        lines.append(f"Required n for {label} effect (d={d}): {n_required}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report
