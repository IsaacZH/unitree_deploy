import numpy as np


class ActionManager:
    def __init__(self, config):
        self.config = config
        self._build_action_definitions()

    def _build_action_definitions(self):
        for _, action_config in self.config.actions.items():
            self.action_clip = action_config["clip"]
            self.action_scale = action_config["scale"]
            break

    def forward(self, policy_action):
        action = policy_action.copy()

        clip_min = np.array([c[0] for c in self.action_clip], dtype=np.float32)
        clip_max = np.array([c[1] for c in self.action_clip], dtype=np.float32)
        action = np.clip(action, clip_min, clip_max)
        action = action * np.array(self.action_scale, dtype=np.float32)

        target_dof_pos = self.config.default_joint_pos + action

        return target_dof_pos
