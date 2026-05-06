import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        self.set_from_key_mask(keys)
        self.set_axes(
            lx=struct.unpack("f", data[4:8])[0],
            rx=struct.unpack("f", data[8:12])[0],
            ry=struct.unpack("f", data[12:16])[0],
            ly=struct.unpack("f", data[20:24])[0],
        )

    def set_axes(self, lx: float, ly: float, rx: float, ry: float):
        self.lx = float(lx)
        self.ly = float(ly)
        self.rx = float(rx)
        self.ry = float(ry)

    def set_from_key_mask(self, keys: int):
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i

    def set_buttons(self, buttons):
        for i in range(16):
            self.button[i] = int(bool(buttons[i])) if i < len(buttons) else 0
