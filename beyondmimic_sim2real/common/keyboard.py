"""Global keyboard listener matching huhai463127310/RoboMimic_Deploy."""

import threading

from pynput import keyboard as pynput_keyboard


class Keyboard:
    """pynput global keyboard; same mapping as the huhai fork."""

    def __init__(self):
        self.key_states = {}
        self.key_prev_states = {}
        self.key_pressed_events = {}
        self.key_released_events = {}
        self._listener = None
        self._lock = threading.Lock()

        self.key_map = {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "0": "0",
            "NUMPAD1": "num1",
            "NUMPAD2": "num2",
            "NUMPAD3": "num3",
            "NUMPAD4": "num4",
            "NUMPAD5": "num5",
            "NUMPAD6": "num6",
            "NUMPAD7": "num7",
            "NUMPAD8": "num8",
            "NUMPAD9": "num9",
            "NUMPAD0": "num0",
            "F1": "f1",
            "F2": "f2",
            "F3": "f3",
            "F4": "f4",
            "F5": "f5",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "E": "e",
            "F": "f",
            "G": "g",
            "H": "h",
            "I": "i",
            "J": "j",
            "K": "k",
            "L": "l",
            "M": "m",
            "N": "n",
            "O": "o",
            "P": "p",
            "Q": "q",
            "R": "r",
            "S": "s",
            "T": "t",
            "U": "u",
            "V": "v",
            "W": "w",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "SPACE": "space",
            "ESCAPE": "esc",
            "ENTER": "enter",
            "TAB": "tab",
            "BACKSPACE": "backspace",
            "LSHIFT": "shift",
            "RSHIFT": "shift",
            "LCTRL": "ctrl_l",
            "RCTRL": "ctrl_r",
            "LALT": "alt_l",
            "RALT": "alt_r",
            "UP": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
        }
        self._start_listener()
        print("Keyboard control (huhai mapping):")
        print("  SPACE=stand  P=passive  Esc=quit")
        print("  Shift+1 / Numpad1 = walk")
        print("  Shift+2 / Numpad2 = dance")
        print("  Shift+3 / Numpad3 = kungfu")
        print("  Shift+4 / Numpad4 = kick")
        print("  Shift+5 / Numpad5 = beyond_mimic")
        print("  Shift+6 / Numpad6 = holomotion")
        print("  Shift+WASD move  Shift+Q/E yaw  arrows move")

    def _normalize_key(self, key) -> str:
        k_str = str(key)
        if "np" in k_str or (k_str.startswith("<") and k_str.endswith(">")):
            try:
                num = int(k_str.strip("<>")) - 96
                if 0 <= num <= 9:
                    return f"num{num}"
            except ValueError:
                pass
            return k_str.replace("Key.", "").lower()
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return str(key).replace("Key.", "").lower()

    def _start_listener(self):
        def on_press(key):
            try:
                k = self._normalize_key(key)
                with self._lock:
                    self.key_states[k] = True
            except Exception as e:
                print(f"[Keyboard] Error on press: {e}")

        def on_release(key):
            try:
                k = self._normalize_key(key)
                with self._lock:
                    self.key_states[k] = False
            except Exception as e:
                print(f"[Keyboard] Error on release: {e}")

        self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def update(self):
        with self._lock:
            self.key_pressed_events.clear()
            self.key_released_events.clear()

            for key_name, key_char in self.key_map.items():
                current = self.key_states.get(key_char, False)
                prev = self.key_prev_states.get(key_char, False)
                if current and not prev:
                    self.key_pressed_events[key_name] = True
                if not current and prev:
                    self.key_released_events[key_name] = True
                self.key_prev_states[key_char] = current

            mapped_chars = set(self.key_map.values())
            for key_char in list(self.key_states.keys()):
                if key_char not in mapped_chars:
                    current = self.key_states.get(key_char, False)
                    prev = self.key_prev_states.get(key_char, False)
                    if current and not prev:
                        self.key_pressed_events[key_char] = True
                    if not current and prev:
                        self.key_released_events[key_char] = True
                    self.key_prev_states[key_char] = current

    def is_key_pressed(self, key):
        key_char = self.key_map.get(key, key.lower())
        with self._lock:
            return self.key_states.get(key_char, False)

    def is_key_released(self, key):
        key_lower = key.lower()
        mapped_key = self.key_map.get(key, key_lower)
        return (
            self.key_released_events.get(key, False)
            or self.key_released_events.get(mapped_key, False)
            or self.key_released_events.get(key_lower, False)
        )

    def get_axis_from_keys(self, neg_key, pos_key):
        neg = self.is_key_pressed(neg_key)
        pos = self.is_key_pressed(pos_key)
        if neg and pos:
            return 0.0
        if neg:
            return -1.0
        if pos:
            return 1.0
        return 0.0
