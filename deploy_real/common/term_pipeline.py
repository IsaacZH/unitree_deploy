import numpy as np


def apply_clip_scale(values, clip, scale):
    values = np.asarray(values, dtype=np.float32).ravel()
    values = np.clip(values, clip[0], clip[1])
    values = values * np.asarray(scale, dtype=np.float32)
    return values


def collect_scaled_terms(raw_state, config, names, clips, scales, value_provider):
    parts = []
    for i, name in enumerate(names):
        values = value_provider(name, raw_state, config)
        values = apply_clip_scale(values, clips[i], scales[i])
        parts.append(values)
    return parts
