import numpy as np

TERMS = {}


def register_term(name):
    def decorator(fn):
        TERMS[name] = fn
        return fn
    return decorator


@register_term("velocity_commands")
def velocity_commands(raw_state, config):
    if hasattr(raw_state, "navigation_manager") and raw_state.navigation_manager.use_navigation_command():
        cmd = raw_state.navigation_manager.get_navigation_velocity_command(raw_state, config)
        cmd = np.asarray(cmd, dtype=np.float32).ravel()
        raw_state.nav_last_action = cmd.copy()
        raw_state.velocity_command = cmd.copy()
        return cmd

    ranges = config.commands["base_velocity"]["ranges"]
    lin_x_max = float(max(abs(ranges["lin_vel_x"][0]), abs(ranges["lin_vel_x"][1])))
    lin_y_max = float(max(abs(ranges["lin_vel_y"][0]), abs(ranges["lin_vel_y"][1])))
    ang_z_max = float(max(abs(ranges["ang_vel_z"][0]), abs(ranges["ang_vel_z"][1])))
    cmd = np.array([
        raw_state.remote.ly * lin_x_max,
        raw_state.remote.lx * -lin_y_max,
        raw_state.remote.rx * -ang_z_max,
    ], dtype=np.float32)
    raw_state.velocity_command = cmd.copy()
    return cmd


@register_term("target_position")
def target_position(raw_state, config):
    nav_manager = raw_state.navigation_manager if hasattr(raw_state, "navigation_manager") else None
    if hasattr(raw_state, "navigation_manager") and raw_state.navigation_manager is not None:
        goal_w = np.asarray(nav_manager.get_target_position(), dtype=np.float32).ravel()
    else:
        goal_w = np.asarray(config.navigation["fixed_target_position"], dtype=np.float32).ravel()

    if goal_w.shape[0] != 3:
        raise ValueError(f"fixed_target_position must be 3D [x, y, z], got shape {goal_w.shape}")

    quat_wxyz = np.asarray(raw_state.imu.quaternion, dtype=np.float32).ravel()[:4]
    if hasattr(raw_state, "high_state") and raw_state.high_state is not None and hasattr(raw_state.high_state, "position"):
        robot_pos_w = np.asarray(raw_state.high_state.position, dtype=np.float32).ravel()[:3]
    else:
        robot_pos_w = np.zeros(3, dtype=np.float32)

    if nav_manager is None:
        from common.navigation_command_manager import NavigationCommandManager

        goal_cmd_body, _ = NavigationCommandManager.compute_goal_command_body(goal_w, robot_pos_w, quat_wxyz)
    else:
        goal_cmd_body, _ = nav_manager.compute_goal_command_body(goal_w, robot_pos_w, quat_wxyz)
    return goal_cmd_body


@register_term("nav_last_action")
def nav_last_action(raw_state, _config):
    if hasattr(raw_state, "nav_last_action") and raw_state.nav_last_action is not None:
        return np.asarray(raw_state.nav_last_action, dtype=np.float32).ravel()
    return np.zeros(3, dtype=np.float32)


@register_term("base_ang_vel")
def base_ang_vel(raw_state, _config):
    return np.asarray(raw_state.imu.gyroscope, dtype=np.float32).ravel()


@register_term("base_lin_vel")
def base_lin_vel(raw_state, _config):
    if hasattr(raw_state, "high_state") and raw_state.high_state is not None:
        return np.asarray(raw_state.high_state.velocity, dtype=np.float32).ravel()
    return np.zeros(3, dtype=np.float32)


@register_term("projected_gravity")
def projected_gravity(raw_state, _config):
    from common.rotation_helper import get_gravity_orientation
    quat = raw_state.imu.quaternion
    return get_gravity_orientation(quat)


@register_term("joint_pos_rel")
def joint_pos_rel(raw_state, config):
    qj = np.zeros(len(config.default_joint_pos), dtype=np.float32)
    for i, joint_id in enumerate(config.joint_ids_map):
        qj[i] = raw_state.motor_state[joint_id].q
    return qj - config.default_joint_pos


@register_term("joint_vel_rel")
def joint_vel_rel(raw_state, config):
    dqj = np.zeros(len(config.default_joint_pos), dtype=np.float32)
    for i, joint_id in enumerate(config.joint_ids_map):
        dqj[i] = raw_state.motor_state[joint_id].dq
    return dqj


@register_term("last_action")
def last_action(raw_state, config):
    return raw_state.last_action.copy() if hasattr(raw_state, 'last_action') else np.zeros(len(config.default_joint_pos), dtype=np.float32)


@register_term("depth_image")
def depth_image(raw_state, _config):
    """Encoded D435i depth feature vector from DepthImageObserver."""
    return raw_state.depth_feature
