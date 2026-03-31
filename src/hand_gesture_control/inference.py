"""Inference utilities for real-time gesture prediction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class PredictionSmoother:
    """Smooths predictions over a sliding window to reduce flickering.

    Uses exponential moving average (EMA) on class probabilities,
    then applies confidence threshold to output stable predictions.
    """

    num_classes: int
    window_size: int = 5
    ema_alpha: float = 0.4  # Higher = more weight on recent predictions
    confidence_threshold: float = 0.6

    # Internal state
    _prob_history: deque = field(default_factory=deque, repr=False)
    _ema_probs: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self):
        self._prob_history = deque(maxlen=self.window_size)
        self._ema_probs = None

    def reset(self) -> None:
        """Clear prediction history."""
        self._prob_history.clear()
        self._ema_probs = None

    def update(self, probs: torch.Tensor) -> tuple[int, float]:
        """Update with new prediction probabilities.

        Args:
            probs: Tensor of shape (num_classes,) with softmax probabilities

        Returns:
            Tuple of (predicted_class_idx, smoothed_confidence)
        """
        probs = probs.detach().cpu()

        # Update EMA
        if self._ema_probs is None:
            self._ema_probs = probs.clone()
        else:
            self._ema_probs = (
                self.ema_alpha * probs +
                (1 - self.ema_alpha) * self._ema_probs
            )

        # Also keep window history for potential future use
        self._prob_history.append(probs)

        # Get prediction from smoothed probabilities
        confidence, idx = torch.max(self._ema_probs, dim=0)

        return int(idx), float(confidence)

    def get_smoothed_probs(self) -> torch.Tensor | None:
        """Return current smoothed probability distribution."""
        return self._ema_probs


@dataclass
class GestureState:
    """Tracks the current gesture state with hold timing."""

    current_gesture: str = "none"
    confidence: float = 0.0
    frames_held: int = 0

    # Thresholds
    min_hold_frames: int = 8  # ~0.27s at 30fps
    confidence_threshold: float = 0.6

    def update(
        self,
        predicted_gesture: str,
        confidence: float
    ) -> tuple[str, bool]:
        """Update state with new prediction.

        Args:
            predicted_gesture: The gesture class name
            confidence: Prediction confidence

        Returns:
            Tuple of (stable_gesture, just_triggered)
            - stable_gesture: The confirmed gesture (or "none")
            - just_triggered: True if gesture just became confirmed this frame
        """
        just_triggered = False

        # Check if confidence meets threshold
        if confidence < self.confidence_threshold:
            predicted_gesture = "none"

        # Same gesture as before?
        if predicted_gesture == self.current_gesture:
            self.frames_held += 1
            self.confidence = confidence
        else:
            # Gesture changed - reset counter
            self.current_gesture = predicted_gesture
            self.confidence = confidence
            self.frames_held = 1

        # Determine stable output
        if self.frames_held >= self.min_hold_frames and self.current_gesture != "none":
            # Check if this is the frame where we just hit the threshold
            if self.frames_held == self.min_hold_frames:
                just_triggered = True
            return self.current_gesture, just_triggered

        return "none", False

    def reset(self) -> None:
        """Reset state."""
        self.current_gesture = "none"
        self.confidence = 0.0
        self.frames_held = 0
