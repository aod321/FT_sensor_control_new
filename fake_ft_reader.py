import time
import numpy as np
import keyboard


class FakeFTReader:
    """ Emulate the FT sensor """

    def __init__(self):
        self.data = np.zeros(6)
        self.keys_add = {
            'q': 0,
            'w': 1,
            'e': 2,
            'r': 3,
            't': 4,
            'y': 5
        }
        self.keys_sub = {
            'a': 0,
            's': 1,
            'd': 2,
            'f': 3,
            'g': 4,
            'h': 5
        }
        self.running = True
        self.setup_key_hooks()

    def setup_key_hooks(self):
        keyboard.on_press(self.on_key_event)

    def on_key_event(self, event):
        key = event.name
        if key in self.keys_add:
            self.data[self.keys_add[key]] += 1
        elif key in self.keys_sub:
            self.data[self.keys_sub[key]] -= 1

    def get_filtered_latest(self):
        return self.data * 1e8

    def warmed_up(self):
        return True

    def run_thread(self):
        while self.running:
            time.sleep(0.01)
