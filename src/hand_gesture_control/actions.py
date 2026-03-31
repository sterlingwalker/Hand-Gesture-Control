"""UI action mapping for gesture control."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, field
from typing import Callable

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.05  # Small delay between actions
    HAS_PYAUTOGUI = True
except ImportError:
    pyautogui = None
    HAS_PYAUTOGUI = False


@dataclass
class ActionMapper:
    """Maps gestures to UI actions.

    Default mappings:
        palm    → switch window (Alt+Tab)
        fist    → Tab key
        like    → Enter key
        dislike → Backspace key
        ok      → Space key
        peace   → close window (Alt+F4)
        one     → open browser (google.com)
    """

    enabled: bool = True
    verbose: bool = True

    # Cooldown tracking (prevent repeated triggers)
    _last_gesture: str = field(default="none", repr=False)
    _cooldown_frames: int = field(default=0, repr=False)
    cooldown_duration: int = 15  # Frames to wait before re-triggering same gesture

    # Custom action overrides
    _custom_actions: dict[str, Callable[[], None]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not HAS_PYAUTOGUI:
            print("Warning: pyautogui not installed. Actions will be simulated.")
            print("Install with: pip install pyautogui")

    def set_action(self, gesture: str, action: Callable[[], None]) -> None:
        """Set a custom action for a gesture."""
        self._custom_actions[gesture] = action

    def execute(self, gesture: str, just_triggered: bool) -> str | None:
        """Execute action for a gesture.

        Args:
            gesture: The gesture name
            just_triggered: True if gesture was just confirmed this frame

        Returns:
            Description of action taken, or None if no action
        """
        if not self.enabled:
            return None

        if not just_triggered:
            return None

        # Handle cooldown
        if gesture == self._last_gesture and self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return None

        # Check for custom action first
        if gesture in self._custom_actions:
            self._custom_actions[gesture]()
            self._last_gesture = gesture
            self._cooldown_frames = self.cooldown_duration
            return f"custom:{gesture}"

        # Default actions
        action_taken = self._execute_default(gesture)

        if action_taken:
            self._last_gesture = gesture
            self._cooldown_frames = self.cooldown_duration

        return action_taken

    def _execute_default(self, gesture: str) -> str | None:
        """Execute default action for gesture."""
        if gesture == "palm":
            # Switch active window (Alt+Tab)
            if HAS_PYAUTOGUI:
                pyautogui.hotkey("alt", "tab")
            if self.verbose:
                print("Action: switch window")
            return "switch window"

        elif gesture == "fist":
            # Tab key
            if HAS_PYAUTOGUI:
                pyautogui.press("tab")
            if self.verbose:
                print("Action: tab")
            return "tab"

        elif gesture == "like":
            # Enter key
            if HAS_PYAUTOGUI:
                pyautogui.press("enter")
            if self.verbose:
                print("Action: enter")
            return "enter"

        elif gesture == "dislike":
            # Backspace key
            if HAS_PYAUTOGUI:
                pyautogui.press("backspace")
            if self.verbose:
                print("Action: backspace")
            return "backspace"

        elif gesture == "ok":
            # Space key
            if HAS_PYAUTOGUI:
                pyautogui.press("space")
            if self.verbose:
                print("Action: space")
            return "space"

        elif gesture == "peace":
            # Close active window (Alt+F4)
            if HAS_PYAUTOGUI:
                pyautogui.hotkey("alt", "F4")
            if self.verbose:
                print("Action: close window")
            return "close window"

        elif gesture == "one":
            # Launch default browser
            webbrowser.open("https://www.google.com")
            if self.verbose:
                print("Action: open browser")
            return "open browser"

        return None

    def tick(self) -> None:
        """Call each frame to update cooldown."""
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1

    def reset(self) -> None:
        """Reset state."""
        self._last_gesture = "none"
        self._cooldown_frames = 0


# Convenience functions for common actions
def click() -> None:
    """Perform mouse click."""
    if HAS_PYAUTOGUI:
        pyautogui.click()


def press_key(key: str) -> None:
    """Press a keyboard key."""
    if HAS_PYAUTOGUI:
        pyautogui.press(key)


def hotkey(*keys: str) -> None:
    """Press a keyboard shortcut."""
    if HAS_PYAUTOGUI:
        pyautogui.hotkey(*keys)


def move_mouse(dx: int, dy: int) -> None:
    """Move mouse by relative amount."""
    if HAS_PYAUTOGUI:
        pyautogui.move(dx, dy)
