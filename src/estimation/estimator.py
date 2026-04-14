"""
Aggregate Response Estimator

Predicts P_agg = f(signal) without device-level feedback, using a two-layer
architecture:

    Layer 1 — Dual Quantile Neural Network:
        Separate charge/discharge NNs predict [q10, q50, q90] of P_agg.
        Input: 10 features (intensity, supply_demand, direction,
               intensity*direction, hour_sin, hour_cos, is_peak,
               is_weekend, region_id, priority).
        Split: charge NN (supply_demand < 8), discharge NN (>= 8).
        Loss: Huber quantile loss with AdamW optimiser.

    Layer 2 — Conformal Quantile Regression (CQR):
        Provides distribution-free prediction intervals with finite-sample
        coverage guarantee >= 1 - alpha (Romano et al., NeurIPS 2019).
        Nonconformity score: max(q10 - y, y - q90).
        Calibration: 30% holdout from training data.

    Online update:
        CQR calibration buffer adapts during deployment (no NN weight update
        in the main experiment loop; NN adaptation is used only for
        cross-region transfer in Result 4).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import math

from .conformal import (
    ConformalPredictor,
    CoverageTracker,
    PredictionInterval,
    CQRPredictor,
)
@dataclass
class EstimationResult:
    """Result of response estimation."""

    # Point estimate of aggregate response (kW)
    response_kw: float

    # Prediction interval bounds (kW)
    lower_bound: float
    upper_bound: float

    # Coverage probability (e.g., 0.9 for 90% PI)
    coverage: float = 0.9

    # Confidence score [0, 1]
    confidence: float = 0.5

    # Timestamp of estimation
    timestamp: float = field(default_factory=time.time)

    # Tail risk metrics (reserved, always None)
    tail_risk: Optional[Any] = None

    # Layer contributions (for diagnostics)
    layer_contributions: Optional[Dict[str, float]] = None

    @property
    def interval_width(self) -> float:
        """Width of the prediction interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (interval width / point estimate)."""
        if abs(self.response_kw) < 1e-6:
            return float('inf')
        return self.interval_width / abs(self.response_kw)

    def is_within_interval(self, actual: float) -> bool:
        """Check if actual value falls within prediction interval."""
        return self.lower_bound <= actual <= self.upper_bound


@dataclass
class EstimatorConfig:
    """Estimator configuration."""

    # Coverage probability for prediction intervals
    target_coverage: float = 0.9

    # Minimum confidence threshold
    min_confidence: float = 0.3

    # Historical window size (seconds)
    history_window: float = 3600.0  # 1 hour

    # Enable conformal prediction
    enable_conformal: bool = True

    # Enable CQR (Conformalized Quantile Regression)
    # When True, uses [q10, q90] from Quantile NN as base interval
    # When False, uses traditional conformal with q50 ± calibrated_error
    use_cqr: bool = True

    # Conformal prediction parameters
    conformal_adaptive: bool = True
    conformal_window_size: int = 100
    conformal_coverage_margin: float = 0.0  # No artificial margin; CQR provides finite-sample coverage guarantee

    # Online learning validation parameters
    online_validation_window: int = 50
    online_validation_patience: int = 3

    # PyTorch training parameters
    use_pytorch: bool = False
    pytorch_epochs: int = 100
    pytorch_batch_size: int = 32
    pytorch_learning_rate: float = 0.001

    # Single-NN mode: train one NN on all data (charge+discharge combined)
    # Used for ablation study to measure dual-NN contribution
    use_single_nn: bool = False


class EPSEstimator:
    """
    Aggregate Response Estimator for EPS.

    Estimates the total device response to broadcast signals without
    requiring individual ACK feedback. Uses a dual-quantile architecture:

    1. Quantile Regression NN: Direct aggregate response prediction
       - Outputs [q10, q50, q90] for uncertainty quantification
       - q50 (median) is more robust to outliers than mean

    2. Conformal Prediction: Distribution-free coverage guarantees
       - Split conformal with absolute residuals
       - Guaranteed finite-sample coverage (PICP >= target)
    """

    def __init__(self, config: Optional[EstimatorConfig] = None):
        """
        Initialize the estimator.

        Args:
            config: Estimator configuration. Uses defaults if None.
        """
        self.config = config or EstimatorConfig()
        self._history: List[Dict[str, Any]] = []
        self._is_fitted = False

        # Online learning validation monitoring
        self._validation_window: List[Dict[str, Any]] = []
        self._validation_errors: List[float] = []
        self._nn_update_frozen = False
        self._validation_deterioration_count = 0

        # Check PyTorch availability
        self._pytorch_available = False
        if self.config.use_pytorch:
            try:
                import torch
                self._pytorch_available = True
            except ImportError:
                self._pytorch_available = False

        # Initialize Conformal predictor
        self._conformal: Optional[ConformalPredictor] = None
        self._cqr: Optional[CQRPredictor] = None
        self._coverage_tracker: Optional[CoverageTracker] = None
        if self.config.enable_conformal:
            self._init_conformal()

        # Initialize aggregate prediction model
        self._aggregate_nn: Optional[Any] = None
        self._is_quantile_model: bool = False
        self._learned_params: Optional[Dict[str, Any]] = None

    def _init_conformal(self) -> None:
        """Initialize conformal predictor with split conformal method."""
        from .conformal import ConformalConfig, ConformalMethod

        conformal_config = ConformalConfig(
            target_coverage=self.config.target_coverage,
            coverage_margin=self.config.conformal_coverage_margin,
            adaptive_window=self.config.conformal_window_size,
            method=ConformalMethod.SPLIT,
            use_difficulty_adjustment=False,
        )

        self._conformal = ConformalPredictor(
            config=conformal_config,
            score_type="absolute",
        )

        # Initialize CQR predictor (uses q10/q90 from Quantile NN)
        if self.config.use_cqr:
            self._cqr = CQRPredictor(config=conformal_config)

        self._coverage_tracker = CoverageTracker(
            window_size=self.config.conformal_window_size,
        )

    def _extract_features(
        self,
        signal: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Extract features from a signal for model input.

        Features (10 total, price field reserved):
        1-4: Base signal features (intensity, supply_demand, region_id, priority)
        5-6: Interaction features (intensity*direction, direction)
        7-10: Time features (hour_sin, hour_cos, is_peak, is_weekend)
        """
        # Base features (normalized)
        intensity = signal.get('intensity', 2048) / 4095.0
        supply_demand = signal.get('supply_demand', 8) / 15.0
        region_id = signal.get('region_id', 0) / 16.0
        priority = signal.get('priority', 8) / 15.0

        # Derived features
        direction = 2.0 * (supply_demand - 0.5)  # [-1, 1]

        # Time features (critical for prediction accuracy)
        hour = signal.get('hour', 12)
        if isinstance(hour, float):
            hour = int(hour) % 24
        else:
            hour = hour % 24

        # Cyclic encoding for hour (preserves continuity: 23:00 -> 00:00)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # Peak hours: morning (7-11) and evening (17-21)
        is_peak = 1.0 if (7 <= hour <= 11) or (17 <= hour <= 21) else 0.0

        # Weekend indicator (from timestamp or day_of_week if available)
        day_of_week = signal.get('day_of_week', None)
        if day_of_week is not None:
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
        else:
            is_weekend = 0.0

        features = [
            # Base features (4)
            intensity, supply_demand, region_id, priority,
            # Interaction features (2)
            intensity * direction, direction,
            # Time features (4)
            hour_sin, hour_cos, is_peak, is_weekend,
        ]

        return features

    def fit(
        self,
        historical_signals: List[Dict[str, Any]],
        historical_responses: List[float],
        context_features: Optional[List[Dict[str, Any]]] = None,
        warm_start_state: Optional[Dict[str, Any]] = None,
        **kwargs,  # Reserved parameters for extended models
    ) -> 'EPSEstimator':
        """
        Fit the estimator on historical data.

        Args:
            historical_signals: List of past broadcast signals
            historical_responses: Corresponding aggregate responses (kW)
            context_features: Optional context (weather, time, etc.)
            warm_start_state: Optional model weights from previous training cycle.
                For incremental/online learning, pass the previous model's state_dict
                to initialize weights instead of random initialization.
                Expected format: {'charge_nn': state_dict, 'discharge_nn': state_dict}
            **kwargs: Ignored for backward compatibility

        Returns:
            self for method chaining
        """
        if len(historical_signals) == 0 or len(historical_responses) == 0:
            raise ValueError("Cannot fit with empty signals or responses")

        if len(historical_signals) != len(historical_responses):
            raise ValueError(
                f"Signal count ({len(historical_signals)}) must match "
                f"response count ({len(historical_responses)})"
            )

        n_samples = len(historical_signals)

        # Store history
        for i, (signal, response) in enumerate(zip(historical_signals, historical_responses)):
            entry = {
                'signal': signal,
                'response': response,
                'context': context_features[i] if context_features else None,
                'timestamp': time.time() - (n_samples - i) * 60,
            }
            self._history.append(entry)

        #  Data split: 70% training, 30% calibration (stratified by direction)
        # This prevents data leakage — the NN never sees calibration data during training.
        # Use fit_count as seed offset so multi-run experiments get different splits.
        import numpy as np
        if not hasattr(self, '_fit_count'):
            self._fit_count = 0
        self._fit_count += 1
        rng_split = np.random.default_rng(42 + self._fit_count)
        pos_idx = [i for i, r in enumerate(historical_responses) if r > 0]
        neg_idx = [i for i, r in enumerate(historical_responses) if r <= 0]
        pos_idx_arr = np.array(pos_idx)
        neg_idx_arr = np.array(neg_idx)
        rng_split.shuffle(pos_idx_arr)
        rng_split.shuffle(neg_idx_arr)

        pos_split = int(len(pos_idx_arr) * 0.7)
        neg_split = int(len(neg_idx_arr) * 0.7)

        train_indices = sorted(pos_idx_arr[:pos_split].tolist() + neg_idx_arr[:neg_split].tolist())
        cal_indices = sorted(pos_idx_arr[pos_split:].tolist() + neg_idx_arr[neg_split:].tolist())

        train_signals = [historical_signals[i] for i in train_indices]
        train_responses = [historical_responses[i] for i in train_indices]
        cal_signals = [historical_signals[i] for i in cal_indices]
        cal_responses = [historical_responses[i] for i in cal_indices]

        # Fit Quantile NN on TRAINING set only (NN never sees calibration data)
        self._fit_regression(train_signals, train_responses, warm_start_state)

        # CQR calibration on HELD-OUT calibration set (no data leakage)
        if self.config.enable_conformal:
            self._fit_conformal_from_split(cal_signals, cal_responses)

        self._is_fitted = True
        return self

    def _fit_regression(
        self,
        signals: List[Dict[str, Any]],
        responses: List[float],
        warm_start_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Fit aggregate response prediction model with direction-specific networks.

        Trains TWO separate Quantile Regression NNs:
        - Charge NN: For positive responses (y >= 0), used in valley filling
        - Discharge NN: For negative responses (y < 0), used in peak shaving

        This separation significantly improves prediction accuracy and
        reduces interval width because each network focuses on a single
        direction with narrower output range.

        Args:
            signals: List of signal dictionaries
            responses: List of response values
            warm_start_state: Optional dict with {'charge_nn': state_dict, 'discharge_nn': state_dict}
                for initializing model weights (incremental learning / warm start)

        """
        if len(signals) < 10:
            self._learned_params = None
            return

        n = len(signals)

        # Build feature matrix
        X = []
        training_history = []
        for i, signal in enumerate(signals):
            features = self._extract_features(signal, history=training_history)
            X.append(features)
            training_history.append({
                'signal': signal,
                'response': responses[i],
            })

        y = list(responses)

        # Try PyTorch Dual Quantile NN
        if self.config.use_pytorch:
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim

                # Define the Quantile NN architecture (shared by both networks)
                class QuantileAggregateNN(nn.Module):
                    """
                    Quantile Regression NN for aggregate response prediction.

                    Architecture balanced for ~2K-5K training samples per direction:
                    - 3 hidden layers (128→64→32) — sufficient capacity for smooth response function
                    - ~12K params — with dropout + weight decay + early stopping, works at 0.3-1.0 data/param ratio
                    - Dropout 0.15 throughout for moderate regularization
                    - No BatchNorm (unstable with small batches <32)
                    """
                    def __init__(self, input_dim):
                        super().__init__()
                        self.shared = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(),
                            nn.Dropout(0.15),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.15),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                        )
                        self.q10_head = nn.Linear(32, 1)
                        self.q50_head = nn.Linear(32, 1)
                        self.q90_head = nn.Linear(32, 1)
                        # Total: 12*128+128 + 128*64+64 + 64*32+32 + 32*3+3 = 1664+8256+2080+99 = 12099

                    def forward(self, x):
                        h = self.shared(x)
                        q10 = self.q10_head(h)
                        q50 = self.q50_head(h)
                        q90 = self.q90_head(h)
                        return torch.cat([q10, q50, q90], dim=1)

                def huber_quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9], weights=None, delta=1.0):
                    """Huber Quantile Loss for quantile regression.

                    Args:
                        pred: Predictions [batch, 3] for q10, q50, q90
                        target: True values [batch, 1]
                        quantiles: Target quantiles [0.1, 0.5, 0.9]
                        weights: Sample weights
                        delta: Huber delta
                    """
                    losses = []
                    for i, q in enumerate(quantiles):
                        pred_q = pred[:, i:i+1]
                        error = target - pred_q
                        abs_error = torch.abs(error)
                        huber_mask = (abs_error <= delta).float()
                        quadratic = 0.5 * error ** 2
                        linear = delta * (abs_error - 0.5 * delta)
                        huber_error = huber_mask * quadratic + (1 - huber_mask) * linear
                        quantile_weight = torch.where(error >= 0, q, 1 - q)
                        loss_q = quantile_weight * huber_error
                        losses.append(loss_q)
                    total_loss = torch.cat(losses, dim=1).mean(dim=1)

                    if weights is not None:
                        total_loss = total_loss * weights

                    return total_loss.mean()

                def train_single_nn(X_data, y_data, input_dim, epochs, batch_size, warm_start=None,
                                    quantiles=[0.1, 0.5, 0.9]):
                    """Train a single Quantile NN with optional warm start.

                    Args:
                        X_data: Feature matrix
                        y_data: Target values
                        input_dim: Input dimension
                        epochs: Training epochs
                        batch_size: Batch size
                        warm_start: Optional state_dict to initialize model weights
                        quantiles: Quantiles to predict [q_low, q_mid, q_high]
                                   Use tighter quantiles like [0.12, 0.5, 0.88] for narrower intervals


                    Returns:
                        model, y_mean, y_std, r2, training_history
                    """
                    if len(X_data) < 10:
                        return None, 0.0, 1.0, 0.0, {'epochs': [], 'train_loss': [], 'val_loss': []}

                    n_samples = len(X_data)

                    # 90/10 train/val split for early stopping
                    val_size = max(int(n_samples * 0.1), 2)
                    train_size = n_samples - val_size

                    import random
                    indices = list(range(n_samples))
                    random.Random(42).shuffle(indices)
                    train_indices = indices[:train_size]
                    val_indices = indices[train_size:]

                    # Normalize targets (train-only to prevent data leakage)
                    train_y = [y_data[i] for i in train_indices]
                    y_mean = sum(train_y) / len(train_y)
                    y_std = (sum((yi - y_mean) ** 2 for yi in train_y) / len(train_y)) ** 0.5
                    if y_std < 1e-6:
                        y_std = 1.0
                    y_normalized = [(yi - y_mean) / y_std for yi in y_data]

                    X_tensor = torch.tensor(X_data, dtype=torch.float32)
                    y_tensor = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)

                    X_train = X_tensor[train_indices]
                    y_train = y_tensor[train_indices]
                    X_val = X_tensor[val_indices]
                    y_val = y_tensor[val_indices]

                    # Sample weights
                    y_abs = torch.tensor([abs(yi) for yi in y_data], dtype=torch.float32)
                    y_abs_median = torch.median(y_abs) if len(y_abs) > 0 else torch.tensor(1.0)
                    sample_weights = 1.0 + torch.log1p(y_abs / (y_abs_median + 1e-6))
                    sample_weights = sample_weights / sample_weights.mean()
                    train_weights = sample_weights[train_indices]

                    model = QuantileAggregateNN(input_dim)

                    # Warm start: load previous model weights for incremental learning
                    # This prevents random initialization fluctuations in online learning
                    use_warm_start = False
                    if warm_start is not None:
                        try:
                            model.load_state_dict(warm_start)
                            use_warm_start = True
                        except Exception:
                            pass  # Ignore if state dict incompatible

                    # Use lower learning rate for warm start to preserve learned weights
                    lr = 0.001 if use_warm_start else 0.003
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=20, T_mult=2, eta_min=1e-5
                    )

                    best_loss = float('inf')
                    best_model_state = None
                    patience = 40
                    patience_counter = 0

                    training_history = {
                        'epochs': [],
                        'train_loss': [],
                        'val_loss': [],
                    }

                    for epoch in range(epochs):
                        # Training phase
                        model.train()
                        train_idx_shuffled = torch.randperm(train_size)
                        total_train_loss = 0.0
                        n_batches = 0

                        for i in range(0, train_size, batch_size):
                            batch_idx = train_idx_shuffled[i:i+batch_size]
                            # Skip batches with <2 samples (BatchNorm1d requires >=2)
                            if len(batch_idx) < 2:
                                continue
                            batch_X = X_train[batch_idx]
                            batch_y = y_train[batch_idx]
                            batch_weights = train_weights[batch_idx]

                            optimizer.zero_grad()
                            pred = model(batch_X)
                            loss = huber_quantile_loss(pred, batch_y, quantiles=quantiles,
                                                       weights=batch_weights, delta=1.0)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                            total_train_loss += loss.item()
                            n_batches += 1

                        avg_train_loss = total_train_loss / max(n_batches, 1)

                        # Validation phase
                        model.eval()
                        with torch.no_grad():
                            # sample weights
                            val_pred = model(X_val)
                            val_loss = huber_quantile_loss(val_pred, y_val, quantiles=quantiles,
                                                           delta=1.0).item()

                        scheduler.step()

                        training_history['epochs'].append(epoch + 1)
                        training_history['train_loss'].append(float(avg_train_loss))
                        training_history['val_loss'].append(float(val_loss))

                        # Early stopping
                        if val_loss < best_loss - 1e-5:
                            best_loss = val_loss
                            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break

                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)

                    # Compute R² for this model
                    model.eval()
                    with torch.no_grad():
                        pred_all = model(X_tensor)
                        pred_normalized = pred_all[:, 1].tolist()
                        predictions = [p * y_std + y_mean for p in pred_normalized]

                    mse = sum((p - a) ** 2 for p, a in zip(predictions, y_data)) / n_samples
                    var_y = sum((yi - y_mean) ** 2 for yi in y_data) / n_samples
                    r2 = 1 - mse / max(var_y, 1e-6)

                    return model, y_mean, y_std, r2, training_history

                input_dim = len(X[0])
                epochs = max(self.config.pytorch_epochs, 150)
                batch_size = min(self.config.pytorch_batch_size, n)

                if self.config.use_single_nn:
                    # Single-NN mode: train one NN on ALL data (charge+discharge)
                    # Used for ablation study to measure dual-NN contribution
                    unified_warm_start = None
                    if warm_start_state is not None:
                        unified_warm_start = warm_start_state.get('charge_nn')

                    unified_nn, unified_y_mean, unified_y_std, unified_r2, unified_history = train_single_nn(
                        X, y, input_dim, epochs, batch_size, unified_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    # Store as both charge and discharge (same model)
                    self._charge_nn = unified_nn
                    self._discharge_nn = unified_nn
                    self._is_dual_model = False
                    self._is_quantile_model = True
                    self._aggregate_nn = unified_nn

                    # Compute overall statistics
                    all_predictions = []
                    if unified_nn:
                        unified_nn.eval()
                        with torch.no_grad():
                            X_tensor_all = torch.tensor(X, dtype=torch.float32)
                            pred_all = unified_nn(X_tensor_all)
                            all_predictions = [
                                pred_all[i, 1].item() * unified_y_std + unified_y_mean
                                for i in range(n)
                            ]
                    else:
                        all_predictions = [0.0] * n

                    mse = sum((p - a) ** 2 for p, a in zip(all_predictions, y)) / n
                    y_mean_all = sum(y) / n
                    var_y = sum((yi - y_mean_all) ** 2 for yi in y) / n
                    r2_overall = 1 - mse / max(var_y, 1e-6)

                    residuals = [p - a for p, a in zip(all_predictions, y)]
                    residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

                    self._learned_params = {
                        'model_type': 'dual_quantile_nn',
                        # Both point to the same unified model
                        'charge_model': unified_nn,
                        'charge_y_mean': unified_y_mean,
                        'charge_y_std': unified_y_std,
                        'charge_r2': unified_r2,
                        'charge_n_samples': n,
                        'discharge_model': unified_nn,
                        'discharge_y_mean': unified_y_mean,
                        'discharge_y_std': unified_y_std,
                        'discharge_r2': unified_r2,
                        'discharge_n_samples': n,
                        # Overall stats
                        'n_samples': n,
                        'r2': r2_overall,
                        'residual_std': residual_std,
                        'quantiles': [0.1, 0.5, 0.9],
                        'charge_training_history': unified_history,
                        'discharge_training_history': unified_history,
                    }
                    return

                else:
                    # Default: Dual-NN mode (split by response direction)
                    charge_indices = [i for i in range(n) if y[i] >= 0]
                    discharge_indices = [i for i in range(n) if y[i] < 0]

                    X_charge = [X[i] for i in charge_indices]
                    y_charge = [y[i] for i in charge_indices]

                    X_discharge = [X[i] for i in discharge_indices]
                    y_discharge = [y[i] for i in discharge_indices]

                    # Extract warm start states for each network
                    charge_warm_start = None
                    discharge_warm_start = None
                    if warm_start_state is not None:
                        charge_warm_start = warm_start_state.get('charge_nn')
                        discharge_warm_start = warm_start_state.get('discharge_nn')

                    charge_nn, charge_y_mean, charge_y_std, charge_r2, charge_history = train_single_nn(
                        X_charge, y_charge, input_dim, epochs, batch_size, charge_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    discharge_nn, discharge_y_mean, discharge_y_std, discharge_r2, discharge_history = train_single_nn(
                        X_discharge, y_discharge, input_dim, epochs, batch_size, discharge_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    # Store both models
                    self._charge_nn = charge_nn
                    self._discharge_nn = discharge_nn
                    self._is_dual_model = True
                    self._is_quantile_model = True

                    # For backward compatibility, also store as _aggregate_nn
                    # (will be used as fallback if direction detection fails)
                    self._aggregate_nn = charge_nn if charge_nn else discharge_nn

                    # Compute overall statistics
                    all_predictions = []
                    for i in range(n):
                        if y[i] >= 0 and charge_nn:
                            model = charge_nn
                            y_mean_local = charge_y_mean
                            y_std_local = charge_y_std
                        elif y[i] < 0 and discharge_nn:
                            model = discharge_nn
                            y_mean_local = discharge_y_mean
                            y_std_local = discharge_y_std
                        else:
                            # Fallback
                            model = self._aggregate_nn
                            y_mean_local = charge_y_mean if charge_nn else discharge_y_mean
                            y_std_local = charge_y_std if charge_nn else discharge_y_std

                        if model:
                            model.eval()
                            with torch.no_grad():
                                x_tensor = torch.tensor([X[i]], dtype=torch.float32)
                                pred = model(x_tensor)
                                pred_val = pred[0, 1].item() * y_std_local + y_mean_local
                                all_predictions.append(pred_val)
                        else:
                            all_predictions.append(0.0)

                    mse = sum((p - a) ** 2 for p, a in zip(all_predictions, y)) / n
                    y_mean_all = sum(y) / n
                    var_y = sum((yi - y_mean_all) ** 2 for yi in y) / n
                    r2_overall = 1 - mse / max(var_y, 1e-6)

                    residuals = [p - a for p, a in zip(all_predictions, y)]
                    residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

                    self._learned_params = {
                        'model_type': 'dual_quantile_nn',
                        # Charge model params
                        'charge_model': charge_nn,
                        'charge_y_mean': charge_y_mean,
                        'charge_y_std': charge_y_std,
                        'charge_r2': charge_r2,
                        'charge_n_samples': len(charge_indices),
                        # Discharge model params
                        'discharge_model': discharge_nn,
                        'discharge_y_mean': discharge_y_mean,
                        'discharge_y_std': discharge_y_std,
                        'discharge_r2': discharge_r2,
                        'discharge_n_samples': len(discharge_indices),
                        # Overall stats
                        'n_samples': n,
                        'r2': r2_overall,
                        'residual_std': residual_std,
                        'quantiles': [0.1, 0.5, 0.9],
                        # Training history
                        'charge_training_history': charge_history,
                        'discharge_training_history': discharge_history,
                    }
                return

            except ImportError:
                pass  # PyTorch not available, fall through to linear regression
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"PyTorch quantile NN training failed: {type(e).__name__}: {e}. "
                    "Falling back to linear regression."
                )
                # Fall through to linear regression

        # Fallback: Linear regression
        X_with_bias = [[1.0] + x for x in X]
        n_features = len(X_with_bias[0])

        # Compute X'X
        XtX = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                XtX[i][j] = sum(X_with_bias[k][i] * X_with_bias[k][j] for k in range(n))

        # Compute X'y
        Xty = [sum(X_with_bias[k][i] * y[k] for k in range(n)) for i in range(n_features)]

        # Add regularization
        lambda_reg = 1e-3
        for i in range(n_features):
            XtX[i][i] += lambda_reg

        weights = self._solve_linear_system(XtX, Xty)

        # Compute training error
        predictions = []
        for x in X_with_bias:
            pred = sum(w * f for w, f in zip(weights, x))
            predictions.append(pred)

        mse = sum((p - a) ** 2 for p, a in zip(predictions, y)) / n
        r2 = 1 - mse / max(sum((r - sum(y)/n) ** 2 for r in y) / n, 1e-6)

        residuals = [p - a for p, a in zip(predictions, y)]
        residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

        self._learned_params = {
            'model_type': 'linear',
            'weights': weights,
            'n_samples': n,
            'mse': mse,
            'r2': r2,
            'response_mean': sum(responses) / n,
            'response_std': (sum((r - sum(responses)/n) ** 2 for r in responses) / n) ** 0.5,
            'residual_std': residual_std,
        }

    def _solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Solve Ax = b using Gaussian elimination with partial pivoting."""
        n = len(b)
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        # Forward elimination
        for i in range(n):
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            if abs(aug[i][i]) < 1e-10:
                continue

            for k in range(i + 1, n):
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                x[i] = 0.0
            else:
                x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i]

        return x

    def _simple_predict(self, signal: Dict[str, Any]) -> float:
        """Predict response using learned regression model."""
        if hasattr(self, '_learned_params') and self._learned_params is not None:
            return self._learned_predict(signal)

        # Point estimate with empirical margin
        intensity = signal.get('intensity', 2048)
        supply_demand = signal.get('supply_demand', 8)

        base = (intensity / 4095) * 100
        sd_factor = 1.0 + max(0, (supply_demand - 8) / 7) * 0.3

        return base * sd_factor

    def _learned_predict(self, signal: Dict[str, Any]) -> float:
        """Make prediction using learned model with direction-specific networks."""
        # Use unified feature extraction
        features = self._extract_features(signal)

        model_type = self._learned_params.get('model_type', 'linear')

        # Dual Quantile NN: select appropriate network based on supply_demand
        if model_type == 'dual_quantile_nn':
            try:
                import torch

                # Determine direction: supply_demand < 8 -> charge NN, >= 8 -> discharge NN
                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    # Valley filling → Charge NN
                    model = self._learned_params.get('charge_model')
                    y_mean = self._learned_params.get('charge_y_mean', 0)
                    y_std = self._learned_params.get('charge_y_std', 1)
                else:
                    # Peak shaving → Discharge NN
                    model = self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('discharge_y_mean', 0)
                    y_std = self._learned_params.get('discharge_y_std', 1)

                if model is None:
                    # Fallback to whichever model exists
                    model = self._learned_params.get('charge_model') or self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('charge_y_mean', self._learned_params.get('discharge_y_mean', 0))
                    y_std = self._learned_params.get('charge_y_std', self._learned_params.get('discharge_y_std', 1))

                if model:
                    model.eval()
                    with torch.no_grad():
                        x = torch.tensor([features], dtype=torch.float32)
                        pred_output = model(x)
                        pred_normalized = pred_output[0, 1].item()  # q50
                        prediction = pred_normalized * y_std + y_mean
                    return prediction
            except Exception:
                pass

        # Linear regression fallback
        if 'weights' in self._learned_params:
            features_with_bias = [1.0] + features
            prediction = sum(w * f for w, f in zip(self._learned_params['weights'], features_with_bias))
            return prediction

        return 0.0

    def _fit_conformal_from_split(
        self,
        cal_signals: List[Dict[str, Any]],
        cal_responses: List[float],
    ) -> None:
        """Fit conformal predictor using pre-split calibration data.

        Receives already-split calibration data that the NN has never seen
        during training, eliminating data leakage between point prediction
        and interval calibration.
        """
        n = len(cal_signals)
        if n < 20:
            return

        cal_predictions = []
        cal_q10_predictions = []
        cal_q90_predictions = []
        cal_actuals = list(cal_responses)

        # Check if we have a Quantile NN for CQR
        has_quantile_nn = (
            (hasattr(self, '_aggregate_nn') and self._aggregate_nn is not None) or
            (hasattr(self, '_charge_nn') and self._charge_nn is not None) or
            (hasattr(self, '_discharge_nn') and self._discharge_nn is not None)
        )
        use_cqr = (
            self.config.use_cqr
            and self._cqr is not None
            and has_quantile_nn
            and getattr(self, '_is_quantile_model', False)
        )

        for signal in cal_signals:
            if use_cqr:
                q10, q50, q90 = self._get_quantile_predictions(signal)
                cal_predictions.append(q50)
                cal_q10_predictions.append(q10)
                cal_q90_predictions.append(q90)
            else:
                pred = self._simple_predict(signal)
                cal_predictions.append(pred)

        # Calibrate standard conformal predictor (as fallback)
        self._conformal.calibrate(
            y_true=cal_actuals,
            y_pred=cal_predictions,
        )

        # Calibrate CQR if available
        if use_cqr and len(cal_q10_predictions) > 0:
            cqr_stats = self._cqr.calibrate(
                y_true=cal_actuals,
                q10_pred=cal_q10_predictions,
                q90_pred=cal_q90_predictions,
            )
            self._cqr_calibration_stats = cqr_stats

        residuals = [abs(a - p) for a, p in zip(cal_actuals, cal_predictions)]
        self._conformal_calibration_stats = {
            'n_calibration': len(cal_actuals),
            'mean_residual': sum(residuals) / len(residuals) if residuals else 0,
            'max_residual': max(residuals) if residuals else 0,
            'median_residual': sorted(residuals)[len(residuals) // 2] if residuals else 0,
            'q90_residual': sorted(residuals)[int(len(residuals) * 0.9)] if residuals else 0,
            'use_cqr': use_cqr,
        }

    def _get_quantile_predictions(self, signal: Dict[str, Any]) -> tuple:
        """
        Get all quantile predictions (q10, q50, q90) from direction-specific Quantile NN.

        Uses separate networks for charge (positive) and discharge (negative) predictions,
        resulting in tighter, more accurate quantile intervals.

        Args:
            signal: The broadcast signal parameters

        Returns:
            Tuple of (q10, q50, q90) predictions
        """
        try:
            import torch

            features = self._extract_features(signal)
            model_type = self._learned_params.get('model_type', 'unknown')

            # Dual Quantile NN: select appropriate network based on supply_demand
            if model_type == 'dual_quantile_nn':
                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    # Valley filling → Charge NN (positive responses)
                    model = self._learned_params.get('charge_model')
                    y_mean = self._learned_params.get('charge_y_mean', 0)
                    y_std = self._learned_params.get('charge_y_std', 1)
                else:
                    # Peak shaving → Discharge NN (negative responses)
                    model = self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('discharge_y_mean', 0)
                    y_std = self._learned_params.get('discharge_y_std', 1)

                if model is None:
                    # Fallback to whichever model exists
                    model = self._learned_params.get('charge_model') or self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('charge_y_mean', self._learned_params.get('discharge_y_mean', 0))
                    y_std = self._learned_params.get('charge_y_std', self._learned_params.get('discharge_y_std', 1))

                if model:
                    model.eval()
                    with torch.no_grad():
                        x = torch.tensor([features], dtype=torch.float32)
                        pred_output = model(x)

                        if pred_output.shape[1] == 3:
                            q10_norm, q50_norm, q90_norm = pred_output[0].tolist()
                            q10 = q10_norm * y_std + y_mean
                            q50 = q50_norm * y_std + y_mean
                            q90 = q90_norm * y_std + y_mean
                            # Enforce quantile ordering: q10 <= q50 <= q90
                            q10, q90 = min(q10, q90), max(q10, q90)
                            q50 = max(q10, min(q50, q90))
                            return (q10, q50, q90)
                        else:
                            pred = pred_output.item() * y_std + y_mean
                            return (pred, pred, pred)


        except Exception:
            pass

        pred = self._simple_predict(signal)
        return (pred, pred, pred)

    def get_training_history(self) -> Dict[str, Any]:
        """Return training history for charge/discharge networks.

        Returns:
            Dict with model_type, per-network epochs, train_loss, val_loss.
        """
        if not self._is_fitted or not hasattr(self, '_learned_params') or not self._learned_params:
            return {
                'model_type': 'not_fitted',
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
            }

        model_type = self._learned_params.get('model_type', 'unknown')

        if model_type == 'dual_quantile_nn':
            charge_history = self._learned_params.get('charge_training_history', {})
            discharge_history = self._learned_params.get('discharge_training_history', {})

            # Use the network with more epochs as the merged timeline
            charge_epochs = charge_history.get('epochs', [])
            discharge_epochs = discharge_history.get('epochs', [])

            if len(charge_epochs) >= len(discharge_epochs):
                merged_epochs = charge_epochs
                merged_train_loss = charge_history.get('train_loss', [])
                merged_val_loss = charge_history.get('val_loss', [])
            else:
                merged_epochs = discharge_epochs
                merged_train_loss = discharge_history.get('train_loss', [])
                merged_val_loss = discharge_history.get('val_loss', [])

            return {
                'model_type': 'dual_quantile_nn',
                'epochs': merged_epochs,
                'train_loss': merged_train_loss,
                'val_loss': merged_val_loss,
                'charge': charge_history,
                'discharge': discharge_history,
            }
        else:
            return {
                'model_type': model_type,
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
            }

    def estimate(
        self,
        signal: Dict[str, Any],
        device_distribution: Optional[Dict[str, int]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,  # Reserved parameters for extended models
    ) -> EstimationResult:
        """
        Estimate aggregate response to a broadcast signal.

        Architecture:
        - Quantile NN: Base prediction (q50 as point estimate)
        - Conformal: Prediction interval with coverage guarantee

        Args:
            signal: The broadcast signal parameters
            device_distribution: Optional device counts by type
            context: Optional context features
            **kwargs: Ignored for backward compatibility

        Returns:
            EstimationResult with point estimate and prediction interval

        Raises:
            RuntimeError: If estimator has not been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Estimator must be fitted before calling estimate(). "
                "Call fit() with training data first."
            )

        layer_contributions = {}

        # Base prediction using direction-specific Quantile NN
        base_prediction = 0.0
        prediction_source = 'none'

        # Option 1: Use trained Quantile NN (dual or single model)
        is_quantile = getattr(self, '_is_quantile_model', False)
        is_dual = getattr(self, '_is_dual_model', False)

        if is_quantile and hasattr(self, '_learned_params') and self._learned_params:
            try:
                # Use unified _get_quantile_predictions which handles dual/single model selection
                q10, q50, q90 = self._get_quantile_predictions(signal)

                base_prediction = q50
                layer_contributions['quantile_q10'] = q10
                layer_contributions['quantile_q50'] = q50
                layer_contributions['quantile_q90'] = q90

                if is_dual:
                    supply_demand = signal.get('supply_demand', 8)
                    if supply_demand < 8:
                        prediction_source = 'dual_quantile_nn_charge'
                        layer_contributions['model_used'] = 'charge_nn'
                    else:
                        prediction_source = 'dual_quantile_nn_discharge'
                        layer_contributions['model_used'] = 'discharge_nn'
                else:
                    prediction_source = 'quantile_nn'

                layer_contributions['aggregate_nn'] = base_prediction

            except Exception:
                pass

        # Option 2: Fallback to learned regression
        if prediction_source == 'none':
            base_prediction = self._simple_predict(signal)
            prediction_source = 'learned_regression'
            layer_contributions['learned_regression'] = base_prediction

        point_estimate = base_prediction

        # Scale by device distribution if provided
        if device_distribution:
            total_devices = sum(device_distribution.values())
            point_estimate *= total_devices / 1000

        layer_contributions['final_source'] = prediction_source

        # Uncertainty quantification: CQR (primary), conformal (secondary), empirical margin
        interval_source = 'empirical'

        # Check if prediction came from a Quantile NN (single or dual model)
        is_quantile_source = prediction_source in (
            'quantile_nn',
            'dual_quantile_nn_charge',
            'dual_quantile_nn_discharge',
        )

        # Option 1: Use CQR (Conformalized Quantile Regression) - uses q10/q90
        if (
            self.config.enable_conformal
            and self.config.use_cqr
            and self._cqr is not None
            and self._cqr._is_calibrated
            and is_quantile_source
        ):
            # Get quantile predictions for CQR
            q10 = layer_contributions.get('quantile_q10', point_estimate)
            q50 = layer_contributions.get('quantile_q50', point_estimate)
            q90 = layer_contributions.get('quantile_q90', point_estimate)

            # Scale quantiles if device distribution provided
            if device_distribution:
                total_devices = sum(device_distribution.values())
                scale_factor = total_devices / 1000
                q10 *= scale_factor
                q90 *= scale_factor

            interval = self._cqr.predict(q10=q10, q50=point_estimate, q90=q90)
            lower_bound = interval.lower_bound
            upper_bound = interval.upper_bound
            confidence = interval.estimated_coverage
            layer_contributions['cqr_adjustment'] = interval.quantile_threshold
            layer_contributions['interval_source'] = 'cqr'
            interval_source = 'cqr'

        # Option 2: Use standard Conformal (q50 ± calibrated_error)
        elif self.config.enable_conformal and self._conformal and self._conformal._is_calibrated:
            interval = self._conformal.predict(point_estimate=point_estimate)
            lower_bound = interval.lower_bound
            upper_bound = interval.upper_bound
            confidence = interval.estimated_coverage
            layer_contributions['conformal'] = interval.quantile_threshold
            layer_contributions['interval_source'] = 'conformal'
            interval_source = 'conformal'

        # Option 3: Empirical margin (when CQR/conformal not available)
        else:
            if hasattr(self, '_learned_params') and self._learned_params is not None:
                residual_std = self._learned_params.get('residual_std', abs(point_estimate) * 0.3)
            else:
                residual_std = abs(point_estimate) * 0.3

            z_score = 1.645 if self.config.target_coverage <= 0.9 else 1.96
            margin = z_score * residual_std

            if hasattr(self, '_learned_params') and self._learned_params is not None:
                r2 = self._learned_params.get('r2', 0)
                if r2 < 0.5:
                    margin *= 1.5
            else:
                margin *= 2.0

            lower_bound = point_estimate - margin
            upper_bound = point_estimate + margin
            confidence = 0.5
            layer_contributions['interval_source'] = 'empirical'

        return EstimationResult(
            response_kw=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            coverage=self.config.target_coverage,
            confidence=confidence,
            layer_contributions=layer_contributions if layer_contributions else None,
        )

    def update(
        self,
        signal: Dict[str, Any],
        actual_response: float,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Update estimator with observed response (online learning).

        Args:
            signal: The broadcast signal that was sent
            actual_response: The observed aggregate response (kW)
            context: Optional context features
            **kwargs: Ignored for backward compatibility
        """
        entry = {
            'signal': signal,
            'response': actual_response,
            'context': context,
            'timestamp': time.time(),
        }
        self._history.append(entry)

        # Prune old history
        cutoff = time.time() - self.config.history_window * 24
        self._history = [h for h in self._history if h['timestamp'] > cutoff]

        # Update conformal predictor
        if self.config.enable_conformal:
            estimated = self.estimate(signal, context=context)

            # Update CQR if available (preferred)
            if self.config.use_cqr and self._cqr and hasattr(self._cqr, 'update'):
                q10, q50, q90 = self._get_quantile_predictions(signal)
                self._cqr.update(
                    y_true=actual_response,
                    q10_pred=q10,
                    q50_pred=q50,
                    q90_pred=q90,
                )
            # Fallback to standard conformal
            elif self._conformal:
                self._conformal.update(estimated.response_kw, actual_response)

            if self._coverage_tracker:
                interval = PredictionInterval(
                    point_estimate=estimated.response_kw,
                    lower_bound=estimated.lower_bound,
                    upper_bound=estimated.upper_bound,
                    target_coverage=self.config.target_coverage,
                )
                self._coverage_tracker.record(interval, actual_response)

    def get_statistics(self) -> Dict[str, Any]:
        """Get estimator statistics."""
        stats = {
            'is_fitted': self._is_fitted,
            'history_size': len(self._history),
            'config': {
                'target_coverage': self.config.target_coverage,
                'enable_conformal': self.config.enable_conformal,
            },
        }

        if self._coverage_tracker:
            stats['coverage'] = {
                'empirical': self._coverage_tracker.get_empirical_coverage(),
                'target': self.config.target_coverage,
                'n_predictions': len(self._coverage_tracker._records),
            }

        if self.config.enable_conformal and self._conformal:
            stats['conformal'] = {
                'is_calibrated': self._conformal._is_calibrated,
                'target_coverage': self._conformal.config.target_coverage,
            }

        return stats

    def reset(self) -> None:
        """Reset estimator state."""
        self._history.clear()
        self._is_fitted = False

        if self._conformal:
            self._conformal.reset()

        if self._coverage_tracker:
            self._coverage_tracker._records.clear()

    def get_layer_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for each layer."""
        diagnostics = {}

        if self.config.enable_conformal and self._conformal:
            diagnostics['conformal'] = {
                'is_calibrated': self._conformal._is_calibrated,
                'target_coverage': self._conformal.config.target_coverage,
                'method': self._conformal.config.method.value,
                'score_type': self._conformal.scorer.score_type,
            }

        return diagnostics

    def _online_update_single_nn(
        self,
        model: Any,
        buffer: List[Dict[str, Any]],
        y_mean: float,
        y_std: float,
        learning_rate: float,
        model_key: str = 'default',
        anchor_lambda: float = 100.0,
    ) -> float:
        """
        Perform a single mini-batch update on one neural network.

        publication-grade online learning with catastrophic forgetting prevention:
        1. Persistent optimizer state (maintains Adam momentum)
        2. L2 weight anchoring toward initial weights
        3. Huber loss for robustness to outliers
        4. Conservative learning rate with warmup

        Args:
            model: The PyTorch model to update
            buffer: List of {signal, response} dictionaries
            y_mean: Mean for response denormalization
            y_std: Std for response denormalization
            learning_rate: Learning rate for this update
            model_key: Key for persistent optimizer ('charge' or 'discharge')

        Returns:
            The loss value after the update
        """
        import torch
        import torch.optim as optim

        # Initialize persistent optimizer storage
        if not hasattr(self, '_online_optimizers'):
            self._online_optimizers = {}
        if not hasattr(self, '_initial_weights'):
            self._initial_weights = {}

        # Save initial weights for regularization (only once)
        if model_key not in self._initial_weights:
            self._initial_weights[model_key] = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }

        # Create or reuse persistent optimizer (maintains momentum state)
        if model_key not in self._online_optimizers:
            self._online_optimizers[model_key] = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01,  # L2 regularization
                betas=(0.9, 0.999),
            )
        optimizer = self._online_optimizers[model_key]

        # Update learning rate if changed
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Build mini-batch
        X_batch = []
        y_batch = []
        for item in buffer:
            features = self._extract_features(item['signal'])
            X_batch.append(features)
            y_normalized = (item['response'] - y_mean) / max(y_std, 1e-6)
            y_batch.append(y_normalized)

        X_tensor = torch.tensor(X_batch, dtype=torch.float32)
        y_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)

        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)

        # Huber Quantile Loss (more robust than pinball)
        quantiles = [0.1, 0.5, 0.9]
        delta = 1.0  # Huber threshold
        losses = []
        for i, q in enumerate(quantiles):
            pred_q = pred[:, i:i+1]
            error = y_tensor - pred_q
            abs_error = torch.abs(error)

            # Huber loss component
            huber = torch.where(
                abs_error <= delta,
                0.5 * error ** 2,
                delta * (abs_error - 0.5 * delta)
            )
            # Quantile weighting
            weight = torch.where(error >= 0, q, 1 - q)
            loss_q = weight * huber
            losses.append(loss_q)

        total_loss = torch.cat(losses, dim=1).mean()

        # L2 weight anchoring: penalize deviation from initial weights
        # (prevents catastrophic forgetting during online learning)
        # Uniform L2 anchoring toward initial weights (not Fisher-weighted EWC).
        # anchor_lambda: default 100 for same-region, use <10 for cross-region transfer
        anchor_loss = 0.0
        for name, param in model.named_parameters():
            if name in self._initial_weights[model_key]:
                anchor_loss += torch.sum((param - self._initial_weights[model_key][name]) ** 2)
        total_loss = total_loss + anchor_lambda * anchor_loss

        total_loss.backward()

        # Conservative gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        model.eval()
        return total_loss.item()

    def _compute_validation_error(self) -> float:
        """Compute mean relative prediction error on the validation window.

        Returns:
            Mean relative error in [0, 1] range.
        """
        if not self._validation_window:
            return 0.0

        errors = []
        for sample in self._validation_window:
            signal = sample['signal']
            actual = sample['response']

            try:
                prediction = self.estimate(signal)
                pred_value = prediction.response_kw
                relative_error = abs(pred_value - actual) / max(abs(actual), 1.0)
                errors.append(relative_error)
            except Exception:
                continue

        if not errors:
            return 0.0

        import numpy as np
        return float(np.mean(errors))

    def online_update(
        self,
        signal: Dict[str, Any],
        actual_response: float,
        learning_rate: float = 0.00005,
        error_threshold: float = 0.15,  # Only update if relative error > 15%
        update_nn: bool = False,  # Default: freeze NN, only update conformal
        mini_batch_size: int = None,  # Override mini-batch size (default: 100 dual / 10 legacy)
        anchor_lambda: float = None,  # L2 anchor strength (default: 100; use <10 for transfer)
        **kwargs,
    ) -> Dict[str, float]:
        """
        Perform selective online learning update with new observation.

        publication-grade online learning strategy (default: conformal-only mode):
        1. Freeze NN weights to preserve learned representations
        2. Online update Conformal calibration to adapt uncertainty estimates
        3. Optionally update NN (set update_nn=True) with L2 weight anchoring

        The key insight is that a well-trained NN generalizes well, but Conformal
        calibration needs adaptation to new data distributions. Updating NN weights
        often leads to catastrophic forgetting.

        Args:
            signal: The broadcast signal that was sent
            actual_response: Actual aggregate response observed
            learning_rate: Learning rate for NN update (only if update_nn=True)
            error_threshold: Only update NN if relative error exceeds this
            update_nn: Whether to update NN weights (default False - recommended)
            **kwargs: Ignored for backward compatibility

        Returns:
            Dict with update statistics
        """
        prediction = self.estimate(signal)
        prediction_error = actual_response - prediction.response_kw

        # Calculate relative error for selective update decision
        relative_error = abs(prediction_error) / max(abs(actual_response), 1.0)

        update_stats = {
            'prediction': prediction.response_kw,
            'actual': actual_response,
            'error': prediction_error,
            'relative_error': relative_error,
            'updated': False,
            'nn_updated': False,
            'charge_nn_updated': False,
            'discharge_nn_updated': False,
            'skipped_low_error': False,
        }

        # Store in history with response for rolling statistics
        self._history.append({
            'signal': signal,
            'response': actual_response,  # Store actual response for rolling stats
            'predicted': prediction.response_kw,
            'actual': actual_response,
            'error': prediction_error,
            'timestamp': time.time(),
        })

        # Update validation window
        self._validation_window.append({
            'signal': signal,
            'response': actual_response,
            'predicted': prediction.response_kw,
        })
        if len(self._validation_window) > self.config.online_validation_window:
            self._validation_window.pop(0)

        # Track low-error samples (but don't skip buffer addition)
        skip_conformal_update = relative_error < error_threshold
        if skip_conformal_update:
            update_stats['skipped_low_error'] = True

        # Initialize online buffers if not exists
        if not hasattr(self, '_online_buffer'):
            self._online_buffer = []  # Legacy single-model buffer
            self._online_update_count = 0
        if not hasattr(self, '_online_buffer_charge'):
            self._online_buffer_charge = []
        if not hasattr(self, '_online_buffer_discharge'):
            self._online_buffer_discharge = []

        # Online learning with catastrophic forgetting prevention:
        # 1. L2 weight anchoring (lambda=100)
        # 2. Conservative learning rate (5e-5) and batch size
        # 3. Validation-based early stopping
        # 4. Automatic NN freeze on sustained validation deterioration

        self._online_update_count += 1

        # Check if using dual NN architecture
        is_dual_model = getattr(self, '_is_dual_model', False)
        model_type = self._learned_params.get('model_type', '') if self._learned_params else ''

        # Only update NN weights if update_nn=True (for transfer learning adaptation)
        # Default (update_nn=False) only updates Conformal calibration to prevent catastrophic forgetting
        if update_nn and not self._nn_update_frozen:
            if is_dual_model and model_type == 'dual_quantile_nn':
                # Dual NN mode: route sample to appropriate buffer based on supply_demand
                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    # Valley filling → Charge buffer
                    self._online_buffer_charge.append({
                        'signal': signal,
                        'response': actual_response,
                    })
                else:
                    # Peak shaving → Discharge buffer
                    self._online_buffer_discharge.append({
                        'signal': signal,
                        'response': actual_response,
                    })

                # Perform mini-batch updates for each model
                # Default 100; override via mini_batch_size parameter for transfer adaptation
                _mbs = mini_batch_size if mini_batch_size is not None else 100
                _al = anchor_lambda if anchor_lambda is not None else 100.0

                try:
                    import torch
                    import torch.optim as optim

                    # Update Charge NN if buffer is full
                    if len(self._online_buffer_charge) >= _mbs:
                        charge_model = self._learned_params.get('charge_model')
                        if charge_model is not None:
                            charge_y_mean = self._learned_params.get('charge_y_mean', 0)
                            charge_y_std = self._learned_params.get('charge_y_std', 1)

                            loss = self._online_update_single_nn(
                                model=charge_model,
                                buffer=self._online_buffer_charge[-_mbs:],
                                y_mean=charge_y_mean,
                                y_std=charge_y_std,
                                learning_rate=learning_rate,
                                model_key='charge',
                                anchor_lambda=_al,
                            )

                            update_stats['charge_nn_updated'] = True
                            update_stats['charge_loss'] = loss
                            update_stats['nn_updated'] = True

                            # Keep recent samples for next batch overlap
                            self._online_buffer_charge = self._online_buffer_charge[-max(32, _mbs // 3):]

                    # Update Discharge NN if buffer is full
                    if len(self._online_buffer_discharge) >= _mbs:
                        discharge_model = self._learned_params.get('discharge_model')
                        if discharge_model is not None:
                            discharge_y_mean = self._learned_params.get('discharge_y_mean', 0)
                            discharge_y_std = self._learned_params.get('discharge_y_std', 1)

                            loss = self._online_update_single_nn(
                                model=discharge_model,
                                buffer=self._online_buffer_discharge[-_mbs:],
                                y_mean=discharge_y_mean,
                                y_std=discharge_y_std,
                                learning_rate=learning_rate,
                                model_key='discharge',
                                anchor_lambda=_al,
                            )

                            update_stats['discharge_nn_updated'] = True
                            update_stats['discharge_loss'] = loss
                            update_stats['nn_updated'] = True

                            # Keep recent samples for next batch overlap
                            self._online_buffer_discharge = self._online_buffer_discharge[-max(32, _mbs // 3):]

                except Exception as e:
                    update_stats['nn_error'] = str(e)

                # Validation-based early stopping for NN updates
                if update_stats['nn_updated'] and len(self._validation_window) >= 20:
                    validation_error = self._compute_validation_error()
                    self._validation_errors.append(validation_error)
                    update_stats['validation_error'] = validation_error

                    if len(self._validation_errors) >= 2:
                        prev_error = self._validation_errors[-2]
                        if validation_error > prev_error * 1.05:
                            self._validation_deterioration_count += 1
                        else:
                            self._validation_deterioration_count = 0

                        # Freeze NN if validation keeps deteriorating
                        if self._validation_deterioration_count >= self.config.online_validation_patience:
                            self._nn_update_frozen = True
                            update_stats['nn_frozen'] = True
                            update_stats['freeze_reason'] = f'validation_error increased {self._validation_deterioration_count} times consecutively'

            else:
                # Legacy single-model mode
                self._online_buffer.append({
                    'signal': signal,
                    'response': actual_response,
                })

                # Perform mini-batch NN update
                _mbs_legacy = mini_batch_size if mini_batch_size is not None else 10
                if (len(self._online_buffer) >= _mbs_legacy and
                    hasattr(self, '_aggregate_nn') and self._aggregate_nn is not None and
                    hasattr(self, '_learned_params') and self._learned_params is not None):

                    try:
                        import torch
                        import torch.optim as optim

                        model = self._aggregate_nn
                        y_mean = self._learned_params.get('y_mean', 0)
                        y_std = self._learned_params.get('y_std', 1)

                        _al_legacy = anchor_lambda if anchor_lambda is not None else 100.0
                        loss = self._online_update_single_nn(
                            model=model,
                            buffer=self._online_buffer[-_mbs_legacy:],
                            y_mean=y_mean,
                            y_std=y_std,
                            learning_rate=learning_rate,
                            model_key='legacy',
                            anchor_lambda=_al_legacy,
                        )

                        update_stats['nn_updated'] = True
                        update_stats['loss'] = loss

                        # Clear buffer after update (keep recent samples for rolling stats)
                        self._online_buffer = self._online_buffer[-5:]

                    except Exception as e:
                        update_stats['nn_error'] = str(e)

        # Update conformal calibration (skip if low error to avoid noise)
        if not skip_conformal_update:
            try:
                # Priority 1: Update CQR if available (needs q10, q50, q90)
                if self.config.use_cqr and self._cqr and hasattr(self._cqr, 'update'):
                    # Get quantile predictions for CQR update
                    q10, q50, q90 = self._get_quantile_predictions(signal)
                    self._cqr.update(
                        y_true=actual_response,
                        q10_pred=q10,
                        q50_pred=q50,
                        q90_pred=q90,
                    )
                    update_stats['updated'] = True
                    update_stats['cqr_updated'] = True
                # Priority 2: Update standard conformal (legacy support)
                elif self._conformal and hasattr(self._conformal, 'update'):
                    self._conformal.update(prediction.response_kw, actual_response)
                    update_stats['updated'] = True
                elif self._conformal and hasattr(self._conformal, 'add_batch'):
                    import numpy as np
                    self._conformal.add_batch(
                        np.array([prediction.response_kw]),
                        np.array([actual_response]),
                    )
                    update_stats['updated'] = True
            except Exception as e:
                update_stats['conformal_update_error'] = str(e)

        # Track coverage
        if self._coverage_tracker and hasattr(self._coverage_tracker, 'record'):
            try:
                interval = PredictionInterval(
                    point_estimate=prediction.response_kw,
                    lower_bound=prediction.lower_bound,
                    upper_bound=prediction.upper_bound,
                    target_coverage=self.config.target_coverage,
                )
                self._coverage_tracker.record(interval, actual_response)
            except Exception:
                pass

        return update_stats

    def get_online_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about online learning performance."""
        if not self._history:
            return {'n_samples': 0}

        import numpy as np

        errors = []
        for record in self._history:
            if 'error' in record:
                errors.append(record['error'])
            elif 'predicted' in record and 'actual' in record:
                errors.append(record['actual'] - record['predicted'])

        if not errors:
            return {'n_samples': len(self._history)}

        errors = np.array(errors)

        stats = {
            'n_samples': len(errors),
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'bias': float(np.mean(errors)),
            'std': float(np.std(errors)),
        }

        if self._coverage_tracker:
            coverage_stats = self._coverage_tracker.get_summary()
            stats['coverage'] = coverage_stats

        return stats

    def save_online_state(self, path: str) -> None:
        """Save online learning state for resumption."""
        import json
        from pathlib import Path

        state = {
            'history': self._history[-1000:],
            'is_fitted': self._is_fitted,
            'stats': self.get_online_learning_stats(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_online_state(self, path: str) -> None:
        """Load online learning state to resume."""
        import json

        with open(path, 'r') as f:
            state = json.load(f)

        self._history = state.get('history', [])
        self._is_fitted = state.get('is_fitted', False)
