from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseState(ABC):
    """Base class for all states in the state machine."""

    def __init__(self, controller):
        """
        Initialize the state with a controller reference.

        Args:
            controller: The main Controller instance
        """
        self.controller = controller
        self.config = controller.config

    @abstractmethod
    def enter(self):
        """Called when entering this state."""
        pass

    @abstractmethod
    def execute(self) -> Optional[str]:
        """
        Execute the state logic.

        Returns:
            The name of the next state to transition to, or None to stay in current state.
        """
        pass

    @abstractmethod
    def exit(self):
        """Called when exiting this state."""
        pass

    def get_state_name(self) -> str:
        """Return the name of this state."""
        return self.__class__.__name__
