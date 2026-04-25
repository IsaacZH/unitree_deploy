import numpy as np

TERMS = {}


def register_term(name):
    def decorator(fn):
        TERMS[name] = fn
        return fn
    return decorator


@register_term("velocity_commands")
def velocity_commands(raw_state, config):
    ranges = config.commands["base_velocity"]["ranges"]
    lin_x_max = float(max(abs(ranges["lin_vel_x"][0]), abs(ranges["lin_vel_x"][1])))
    lin_y_max = float(max(abs(ranges["lin_vel_y"][0]), abs(ranges["lin_vel_y"][1])))
    ang_z_max = float(max(abs(ranges["ang_vel_z"][0]), abs(ranges["ang_vel_z"][1])))
    cmd = np.array([
        raw_state.remote.ly * lin_x_max,
        raw_state.remote.lx * -lin_y_max,
        raw_state.remote.rx * -ang_z_max,
    ], dtype=np.float32)
    return cmd


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
