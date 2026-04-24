import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.weak_motor = config.get("weak_motor", [])

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)
            self.joint_ids_map = config["joint_ids_map"]

            # Manager-based config
            self.use_encoder = config["use_encoder"]
            self.stiffness = config["stiffness"]
            self.damping = config["damping"]

            # Commands, actions, encoder_input, policy_input from YAML
            self.commands = config["commands"]
            self.actions = config["actions"]
            self.encoder_input = config["encoder_input"]
            self.policy_input = config["policy_input"]

            # FixStand config
            self.fixstand_kp = config["FixStand"]["kp"]
            self.fixstand_kd = config["FixStand"]["kd"]
            self.fixstand_ts = config["FixStand"]["ts"]
            self.fixstand_qs = config["FixStand"]["qs"]
            
