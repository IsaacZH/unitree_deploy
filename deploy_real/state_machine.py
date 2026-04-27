from typing import Optional, Dict, Type
from states import BaseState


class StateMachine:
    """Manages state transitions and execution."""

    def __init__(self, controller, initial_state_name: str = "move_to_default_pos"):
        """
        Initialize the state machine.

        Args:
            controller: The main Controller instance
            initial_state_name: Name of the initial state to enter
        """
        self.controller = controller
        self.states: Dict[str, BaseState] = {}
        self.current_state: Optional[BaseState] = None
        self.is_running = True

        # Register all states
        self._register_states()

        # Set initial state
        self.switch_to_state(initial_state_name)

    def _register_states(self):
        """Register all available states."""
        from states import ZeroTorqueState, MoveToDefaultPosState, MoveToSitPosState, RunState

        self.states["zero_torque"] = ZeroTorqueState(self.controller)
        self.states["move_to_default_pos"] = MoveToDefaultPosState(self.controller)
        self.states["move_to_sit_pos"] = MoveToSitPosState(self.controller)
        self.states["run"] = RunState(self.controller)

    def switch_to_state(self, state_name: str):
        """
        Switch to a new state.

        Args:
            state_name: Name of the state to switch to
        """
        if state_name == "exit":
            self.is_running = False
            return

        if state_name not in self.states:
            raise ValueError(f"Unknown state: {state_name}")

        if self.current_state is not None:
            self.current_state.exit()

        self.current_state = self.states[state_name]
        self.current_state.enter()

    def execute(self):
        """Execute one step of the current state."""
        if not self.is_running or self.current_state is None:
            return False

        self.controller.update_control_input()
        next_state = self.current_state.execute()

        if next_state is not None:
            self.switch_to_state(next_state)

        return self.is_running

    def get_current_state(self) -> str:
        """Get the name of the current state."""
        return self.current_state.get_state_name() if self.current_state else "None"
