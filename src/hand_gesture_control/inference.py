"""Inference utilities for real-time gesture prediction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

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


SwipeDirection = Literal["swipe_up", "swipe_down", "swipe_left", "swipe_right"]


@dataclass
class SwipeResult:
    """Result of a swipe detection."""
    direction: SwipeDirection
    velocity: float  # Speed of the swipe (0-1 normalized)


@dataclass
class SwipeDetector:
    """Detects swipe gestures from hand landmark movement.

    Tracks hand position over frames and detects rapid directional movement.
    Requires hand to be still before detecting a new swipe to prevent
    "reset" movements from triggering.
    """

    # Detection thresholds
    min_velocity: float = 0.15  # Minimum movement per frame (as fraction of frame)
    min_frames: int = 3  # Minimum frames of consistent movement
    max_frames: int = 10  # Maximum frames to consider for swipe
    cooldown_frames: int = 20  # Frames to wait after detecting a swipe
    stillness_threshold: float = 0.02  # Max movement to be considered "still"
    stillness_frames: int = 5  # Frames hand must be still before new swipe

    # Internal state
    _position_history: deque = field(default_factory=deque, repr=False)
    _cooldown: int = field(default=0, repr=False)
    _last_swipe: SwipeDirection = field(default=None, repr=False)
    _still_counter: int = field(default=0, repr=False)
    _last_position: tuple[float, float] | None = field(default=None, repr=False)
    _armed: bool = field(default=True, repr=False)  # Ready to detect swipe

    def __post_init__(self):
        self._position_history = deque(maxlen=self.max_frames)
        self._cooldown = 0
        self._last_swipe = None

    def update(self, hand_center: tuple[float, float] | None) -> SwipeResult | None:
        """Update with new hand position and check for swipe.

        Args:
            hand_center: Normalized (x, y) position of hand center (0-1 range),
                        or None if no hand detected.

        Returns:
            SwipeResult with direction and velocity, or None if no swipe.
        """
        # Handle cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            if hand_center is None:
                self._position_history.clear()
                self._last_position = None
            return None

        # No hand - clear history and reset
        if hand_center is None:
            self._position_history.clear()
            self._last_position = None
            self._armed = True  # Re-arm when hand leaves
            self._still_counter = 0
            return None

        # Check if hand is still (for arming swipe detection)
        if self._last_position is not None:
            move_dist = ((hand_center[0] - self._last_position[0])**2 +
                        (hand_center[1] - self._last_position[1])**2)**0.5

            if move_dist < self.stillness_threshold:
                self._still_counter += 1
                if self._still_counter >= self.stillness_frames:
                    self._armed = True  # Hand has been still, ready for swipe
                    self._position_history.clear()  # Clear old movement
            else:
                self._still_counter = 0

        self._last_position = hand_center

        # Only track for swipe if armed
        if not self._armed:
            return None

        # Add to history
        self._position_history.append(hand_center)

        # Need enough history
        if len(self._position_history) < self.min_frames:
            return None

        # Calculate movement over recent frames
        result = self._detect_swipe()

        if result:
            self._cooldown = self.cooldown_frames
            self._last_swipe = result.direction
            self._position_history.clear()
            self._armed = False  # Disarm until hand is still again

        return result

    def _detect_swipe(self) -> SwipeResult | None:
        """Analyze position history to detect swipe."""
        if len(self._position_history) < self.min_frames:
            return None

        # Get recent positions
        positions = list(self._position_history)

        # Calculate total displacement
        start_x, start_y = positions[0]
        end_x, end_y = positions[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Calculate velocity (displacement per frame)
        num_frames = len(positions) - 1
        vel_x = dx / num_frames if num_frames > 0 else 0
        vel_y = dy / num_frames if num_frames > 0 else 0

        abs_vel_x = abs(vel_x)
        abs_vel_y = abs(vel_y)

        # Require dominant direction to be at least 1.5x the other direction
        # This prevents diagonal movements from triggering random swipes
        ratio_threshold = 1.5

        # Check horizontal swipe
        if abs_vel_x > self.min_velocity and abs_dx > abs_dy * ratio_threshold:
            direction = "swipe_right" if dx > 0 else "swipe_left"
            velocity = min(abs_vel_x / 0.10, 1.0)  # Easier to reach max velocity
            return SwipeResult(direction=direction, velocity=velocity)

        # Check vertical swipe
        elif abs_vel_y > self.min_velocity and abs_dy > abs_dx * ratio_threshold:
            direction = "swipe_down" if dy > 0 else "swipe_up"
            velocity = min(abs_vel_y / 0.10, 1.0)  # Easier to reach max velocity
            return SwipeResult(direction=direction, velocity=velocity)

        return None

    def _check_consistency(self, values: list[float], increasing: bool) -> bool:
        """Check if movement is consistently in one direction."""
        if len(values) < 2:
            return True

        consistent_count = 0
        for i in range(1, len(values)):
            diff = values[i] - values[i-1]
            if (increasing and diff > 0) or (not increasing and diff < 0):
                consistent_count += 1

        # Require at least 60% consistency
        return consistent_count >= (len(values) - 1) * 0.6

    def reset(self) -> None:
        """Reset state."""
        self._position_history.clear()
        self._cooldown = 0
        self._last_swipe = None
        self._still_counter = 0
        self._last_position = None
        self._armed = True


@dataclass
class CursorController:
    """Controls mouse cursor based on hand position.

    Uses ABSOLUTE positioning: hand position in frame maps to cursor position on screen.
    Buffer zone shrinks the active tracking area so you don't need to reach frame edges.
    """

    # Screen dimensions (will be set on activation)
    screen_width: int = 1920
    screen_height: int = 1080

    # Smoothing (0-1, higher = smoother but more latency)
    smoothing: float = 0.3

    # Buffer zones - percentage of frame edge to exclude
    buffer_top: float = 0.20
    buffer_bottom: float = 0.12
    buffer_left: float = 0.12
    buffer_right: float = 0.12

    # Internal state
    _active: bool = field(default=False, repr=False)
    _smoothed_pos: tuple[float, float] | None = field(default=None, repr=False)

    def is_active(self) -> bool:
        """Check if cursor control is active."""
        return self._active

    def activate(self, position: tuple[float, float]) -> None:
        """Enter cursor control mode."""
        self._active = True
        self._smoothed_pos = position if position else None
        # Try to get actual screen size
        try:
            import pyautogui
            self.screen_width, self.screen_height = pyautogui.size()
        except:
            pass

    def deactivate(self) -> None:
        """Exit cursor control mode."""
        self._active = False
        self._smoothed_pos = None

    def update(self, position: tuple[float, float] | None) -> tuple[int, int] | None:
        """Update cursor to absolute position based on hand location.

        Args:
            position: Normalized (x, y) hand position (0-1 range), or None.

        Returns:
            (x, y) absolute screen coordinates, or None if not active.
        """
        if not self._active or position is None:
            return None

        # Apply buffer zones - remap position from [buffer_min, 1-buffer_max] to [0, 1]
        def remap(val: float, buf_min: float, buf_max: float) -> float:
            # Clamp to buffer zone, then scale to 0-1
            clamped = max(buf_min, min(1.0 - buf_max, val))
            return (clamped - buf_min) / (1.0 - buf_min - buf_max)

        mapped_x = remap(position[0], self.buffer_left, self.buffer_right)
        mapped_y = remap(position[1], self.buffer_top, self.buffer_bottom)

        # Apply smoothing
        if self._smoothed_pos is None:
            self._smoothed_pos = (mapped_x, mapped_y)
        else:
            smooth_x = self.smoothing * self._smoothed_pos[0] + (1 - self.smoothing) * mapped_x
            smooth_y = self.smoothing * self._smoothed_pos[1] + (1 - self.smoothing) * mapped_y
            self._smoothed_pos = (smooth_x, smooth_y)

        # Convert to screen coordinates
        # Invert X because webcam is mirrored
        screen_x = int((1.0 - self._smoothed_pos[0]) * self.screen_width)
        screen_y = int(self._smoothed_pos[1] * self.screen_height)

        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return (screen_x, screen_y)

    def reset(self) -> None:
        """Reset state."""
        self._active = False
        self._smoothed_pos = None
