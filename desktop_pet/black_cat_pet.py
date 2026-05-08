"""A tiny desktop pet inspired by soft black-cat anime mascots.

Run:
    python desktop_pet/black_cat_pet.py

The pet is borderless, always on top, draggable, and animated with Tkinter.
It uses only the Python standard library.
"""

from __future__ import annotations

import math
import random
import tkinter as tk
from ctypes import windll
from dataclasses import dataclass
from enum import Enum


WIDTH = 260
HEIGHT = 240
TRANSPARENT_COLOR = "#ff00ff"


class Mood(str, Enum):
    IDLE = "idle"
    BLINK = "blink"
    TAIL_SWISH = "tail_swish"
    EAR_TWITCH = "ear_twitch"
    STRETCH = "stretch"
    NAP = "nap"
    EXCITED = "excited"
    SHY = "shy"
    POUNCE = "pounce"


@dataclass
class Bubble:
    text: str
    frames_left: int


class BlackCatPet:
    """Small animated desktop pet drawn with canvas primitives."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Pocket Shadow Cat")
        self.root.geometry(f"{WIDTH}x{HEIGHT}+980+540")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=TRANSPARENT_COLOR)

        try:
            self.root.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            # The fallback still works, just without transparent window corners.
            pass

        self.canvas = tk.Canvas(
            self.root,
            width=WIDTH,
            height=HEIGHT,
            bg=TRANSPARENT_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.frame = 0
        self.mood = Mood.IDLE
        self.mood_frames_left = 0
        self.bubble: Bubble | None = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.drag_start_screen_x = 0
        self.drag_start_screen_y = 0
        self.click_count = 0
        self.next_idle_action = random.randint(70, 150)
        self.walking_enabled = True
        self.walk_frames_left = random.randint(110, 260)
        self.walk_direction = random.choice([-1, 1])

        self.root.bind("<ButtonPress-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._drag)
        self.root.bind("<ButtonRelease-1>", self._click)
        self.root.bind("<Button-3>", self._show_menu)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self.menu = tk.Menu(self.root, tearoff=False)
        self.menu.add_command(label="Pet", command=self._pet)
        self.menu.add_command(label="Snack", command=self._snack)
        self.menu.add_command(label="Nap", command=self._nap)
        self.menu.add_command(label="Toggle stroll", command=self._toggle_stroll)
        self.menu.add_separator()
        self.menu.add_command(label="Exit", command=self.root.destroy)

        self._tick()

    def run(self) -> None:
        self.root.mainloop()

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_offset_x = event.x
        self.drag_offset_y = event.y
        self.drag_start_screen_x = self.root.winfo_pointerx()
        self.drag_start_screen_y = self.root.winfo_pointery()

    def _drag(self, event: tk.Event) -> None:
        x = self.root.winfo_pointerx() - self.drag_offset_x
        y = self.root.winfo_pointery() - self.drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    def _click(self, event: tk.Event) -> None:
        moved_x = abs(self.root.winfo_pointerx() - self.drag_start_screen_x)
        moved_y = abs(self.root.winfo_pointery() - self.drag_start_screen_y)
        if moved_x > 4 or moved_y > 4:
            return

        self.click_count += 1
        if self.click_count % 5 == 0:
            self._set_mood(Mood.POUNCE, 44)
            self._say("zoom!")
        else:
            choice = random.choice(
                [
                    (Mood.EXCITED, "meow!"),
                    (Mood.SHY, "pat pat"),
                    (Mood.EAR_TWITCH, "hmm?"),
                    (Mood.TAIL_SWISH, "purr..."),
                ]
            )
            self._set_mood(choice[0], 46)
            self._say(choice[1])

    def _show_menu(self, event: tk.Event) -> None:
        self.menu.tk_popup(event.x_root, event.y_root)

    def _pet(self) -> None:
        self._set_mood(Mood.EXCITED, 60)
        self._say("happy!")

    def _snack(self) -> None:
        self._set_mood(Mood.POUNCE, 70)
        self._say("snack?")

    def _nap(self) -> None:
        self._set_mood(Mood.NAP, 150)
        self._say("zzz...")

    def _toggle_stroll(self) -> None:
        self.walking_enabled = not self.walking_enabled
        self._say("stroll on" if self.walking_enabled else "stroll off")

    def _set_mood(self, mood: Mood, frames: int) -> None:
        self.mood = mood
        self.mood_frames_left = frames

    def _say(self, text: str, frames: int = 72) -> None:
        self.bubble = Bubble(text=text, frames_left=frames)

    def _maybe_idle_action(self) -> None:
        if self.mood != Mood.IDLE:
            return

        self.next_idle_action -= 1
        if self.next_idle_action > 0:
            return

        mood, frames = random.choice(
            [
                (Mood.BLINK, 18),
                (Mood.TAIL_SWISH, 56),
                (Mood.EAR_TWITCH, 36),
                (Mood.STRETCH, 64),
                (Mood.NAP, 92),
            ]
        )
        self._set_mood(mood, frames)
        self.next_idle_action = random.randint(90, 190)

    def _tick(self) -> None:
        self.frame += 1
        if self.mood_frames_left > 0:
            self.mood_frames_left -= 1
        elif self.mood != Mood.IDLE:
            self.mood = Mood.IDLE

        if self.bubble is not None:
            self.bubble.frames_left -= 1
            if self.bubble.frames_left <= 0:
                self.bubble = None

        self._maybe_idle_action()
        self._maybe_stroll()
        self._draw()
        self.root.after(33, self._tick)

    def _maybe_stroll(self) -> None:
        if not self.walking_enabled or self.mood in {Mood.POUNCE, Mood.NAP}:
            return

        self.walk_frames_left -= 1
        if self.walk_frames_left <= 0:
            self.walk_frames_left = random.randint(110, 260)
            if random.random() < 0.55:
                self.walk_direction *= -1
            return

        if self.walk_frames_left % 3 != 0:
            return

        x = self.root.winfo_x() + self.walk_direction
        y = self.root.winfo_y()
        max_x = self._screen_width() - WIDTH
        if x <= 8 or x >= max_x - 8:
            self.walk_direction *= -1
            x = max(8, min(max_x - 8, x))
        self.root.geometry(f"+{x}+{y}")

    def _screen_width(self) -> int:
        try:
            return windll.user32.GetSystemMetrics(0)
        except (AttributeError, OSError):
            return self.root.winfo_screenwidth()

    def _draw(self) -> None:
        self.canvas.delete("all")

        bounce = math.sin(self.frame / 15) * 2.0
        body_scale_x = 1.0
        body_scale_y = 1.0
        y_shift = bounce
        ear_left_shift = 0.0
        ear_right_shift = 0.0
        tail_swing = math.sin(self.frame / 7) * 8
        eyes_open = 1.0
        mouth = "soft"

        if self.mood == Mood.BLINK:
            eyes_open = 0.08 if 5 < self.mood_frames_left < 14 else 0.7
        elif self.mood == Mood.TAIL_SWISH:
            tail_swing = math.sin(self.frame / 3) * 24
        elif self.mood == Mood.EAR_TWITCH:
            ear_left_shift = math.sin(self.frame / 2.8) * 6
            ear_right_shift = math.cos(self.frame / 3.2) * 4
        elif self.mood == Mood.STRETCH:
            progress = math.sin((64 - self.mood_frames_left) / 64 * math.pi)
            body_scale_x = 1.0 + progress * 0.16
            body_scale_y = 1.0 - progress * 0.10
            y_shift += progress * 10
            eyes_open = 0.55
        elif self.mood == Mood.NAP:
            eyes_open = 0.02
            mouth = "sleep"
            y_shift += math.sin(self.frame / 18) * 1.2 + 6
            tail_swing = math.sin(self.frame / 16) * 6
        elif self.mood == Mood.EXCITED:
            y_shift += math.sin(self.frame / 3) * 6 - 4
            eyes_open = 1.12
            mouth = "smile"
            tail_swing = math.sin(self.frame / 2.5) * 28
            ear_left_shift = -4
            ear_right_shift = 4
        elif self.mood == Mood.SHY:
            y_shift += 8
            eyes_open = 0.62
            mouth = "shy"
            tail_swing = -18
        elif self.mood == Mood.POUNCE:
            phase = (44 - self.mood_frames_left) / 44
            hop = -math.sin(phase * math.pi) * 34
            y_shift += hop
            body_scale_x = 0.92 + math.sin(phase * math.pi) * 0.10
            body_scale_y = 1.06 - math.sin(phase * math.pi) * 0.08
            eyes_open = 1.18
            mouth = "smile"
            tail_swing = math.sin(self.frame / 2.0) * 34

        self._draw_shadow(y_shift)
        self._draw_tail(y_shift, tail_swing)
        self._draw_body(y_shift, body_scale_x, body_scale_y)
        self._draw_ears(y_shift, ear_left_shift, ear_right_shift)
        self._draw_face(y_shift, eyes_open, mouth)
        self._draw_paws(y_shift)
        self._draw_accents(y_shift)

        if self.bubble:
            self._draw_bubble(self.bubble.text)

        if self.mood == Mood.NAP:
            self._draw_sleep_marks()

    def _draw_shadow(self, y_shift: float) -> None:
        self.canvas.create_oval(70, 204 + y_shift, 190, 225 + y_shift, fill="#111111", outline="")
        self.canvas.create_oval(84, 207 + y_shift, 176, 221 + y_shift, fill="#242424", outline="")

    def _draw_tail(self, y_shift: float, swing: float) -> None:
        base_x = 178
        base_y = 152 + y_shift
        self.canvas.create_line(
            base_x,
            base_y,
            213 + swing,
            137 + y_shift,
            207 + swing * 0.6,
            94 + y_shift,
            176 + swing * 0.25,
            84 + y_shift,
            width=25,
            smooth=True,
            fill="#101013",
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )
        self.canvas.create_line(
            base_x + 2,
            base_y - 1,
            211 + swing,
            135 + y_shift,
            205 + swing * 0.6,
            96 + y_shift,
            width=11,
            smooth=True,
            fill="#1d1d22",
            capstyle=tk.ROUND,
        )

    def _draw_body(self, y_shift: float, scale_x: float, scale_y: float) -> None:
        cx, cy = 130, 146 + y_shift
        rx, ry = 58 * scale_x, 67 * scale_y
        self.canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, fill="#101013", outline="")
        self.canvas.create_oval(
            cx - rx + 13,
            cy - ry + 12,
            cx + rx - 15,
            cy + ry - 15,
            fill="#17171d",
            outline="",
        )
        self.canvas.create_oval(87, 130 + y_shift, 173, 198 + y_shift, fill="#202026", outline="")

    def _draw_ears(self, y_shift: float, left_shift: float, right_shift: float) -> None:
        self.canvas.create_polygon(
            86 + left_shift,
            84 + y_shift,
            104 + left_shift,
            25 + y_shift,
            128 + left_shift * 0.2,
            88 + y_shift,
            fill="#101013",
            outline="",
        )
        self.canvas.create_polygon(
            174 + right_shift,
            84 + y_shift,
            156 + right_shift,
            25 + y_shift,
            132 + right_shift * 0.2,
            88 + y_shift,
            fill="#101013",
            outline="",
        )
        self.canvas.create_polygon(
            101 + left_shift,
            75 + y_shift,
            107 + left_shift,
            43 + y_shift,
            120 + left_shift * 0.2,
            82 + y_shift,
            fill="#2b202b",
            outline="",
        )
        self.canvas.create_polygon(
            159 + right_shift,
            75 + y_shift,
            153 + right_shift,
            43 + y_shift,
            140 + right_shift * 0.2,
            82 + y_shift,
            fill="#2b202b",
            outline="",
        )

    def _draw_face(self, y_shift: float, eyes_open: float, mouth: str) -> None:
        self.canvas.create_oval(74, 54 + y_shift, 186, 159 + y_shift, fill="#111116", outline="")
        self.canvas.create_oval(87, 64 + y_shift, 173, 149 + y_shift, fill="#19191f", outline="")
        self._draw_eye(109, 105 + y_shift, eyes_open)
        self._draw_eye(151, 105 + y_shift, eyes_open)

        self.canvas.create_oval(126, 123 + y_shift, 134, 130 + y_shift, fill="#212128", outline="")

        if mouth == "sleep":
            self.canvas.create_arc(117, 130 + y_shift, 143, 141 + y_shift, start=0, extent=180, style=tk.ARC, outline="#d8eef0", width=2)
        elif mouth == "smile":
            self.canvas.create_arc(116, 126 + y_shift, 130, 144 + y_shift, start=260, extent=170, style=tk.ARC, outline="#d8eef0", width=2)
            self.canvas.create_arc(130, 126 + y_shift, 144, 144 + y_shift, start=110, extent=170, style=tk.ARC, outline="#d8eef0", width=2)
        elif mouth == "shy":
            self.canvas.create_line(122, 137 + y_shift, 138, 137 + y_shift, fill="#d8eef0", width=2, smooth=True)
        else:
            self.canvas.create_arc(119, 128 + y_shift, 130, 141 + y_shift, start=260, extent=140, style=tk.ARC, outline="#d8eef0", width=2)
            self.canvas.create_arc(130, 128 + y_shift, 141, 141 + y_shift, start=100, extent=140, style=tk.ARC, outline="#d8eef0", width=2)

    def _draw_eye(self, cx: int, cy: float, openness: float) -> None:
        h = max(2, 22 * openness)
        self.canvas.create_oval(cx - 13, cy - h / 2, cx + 13, cy + h / 2, fill="#91f5ef", outline="")
        if h > 5:
            self.canvas.create_oval(cx - 7, cy - h / 2 + 3, cx + 6, cy + h / 2 - 2, fill="#172023", outline="")
            self.canvas.create_oval(cx - 3, cy - h / 2 + 5, cx + 2, cy - h / 2 + 10, fill="#f7ffff", outline="")
        else:
            self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#91f5ef", width=2, capstyle=tk.ROUND)

    def _draw_paws(self, y_shift: float) -> None:
        self.canvas.create_oval(84, 170 + y_shift, 116, 201 + y_shift, fill="#111116", outline="")
        self.canvas.create_oval(144, 170 + y_shift, 176, 201 + y_shift, fill="#111116", outline="")
        self.canvas.create_oval(96, 188 + y_shift, 104, 195 + y_shift, fill="#31313a", outline="")
        self.canvas.create_oval(156, 188 + y_shift, 164, 195 + y_shift, fill="#31313a", outline="")

    def _draw_accents(self, y_shift: float) -> None:
        if self.mood == Mood.SHY:
            self.canvas.create_oval(84, 120 + y_shift, 102, 131 + y_shift, fill="#573047", outline="")
            self.canvas.create_oval(158, 120 + y_shift, 176, 131 + y_shift, fill="#573047", outline="")
        elif self.mood == Mood.EXCITED:
            self.canvas.create_line(72, 84 + y_shift, 59, 75 + y_shift, fill="#91f5ef", width=3, capstyle=tk.ROUND)
            self.canvas.create_line(188, 84 + y_shift, 201, 75 + y_shift, fill="#91f5ef", width=3, capstyle=tk.ROUND)

    def _draw_bubble(self, text: str) -> None:
        x1, y1, x2, y2 = 23, 20, 92, 55
        self.canvas.create_oval(x1, y1, x2, y2, fill="#f7ffff", outline="#9fdad8", width=2)
        self.canvas.create_polygon(82, 45, 99, 57, 77, 55, fill="#f7ffff", outline="#9fdad8")
        self.canvas.create_text(58, 38, text=text, fill="#202026", font=("Segoe UI", 10, "bold"))

    def _draw_sleep_marks(self) -> None:
        lift = math.sin(self.frame / 10) * 4
        self.canvas.create_text(193, 58 + lift, text="z", fill="#baf8f6", font=("Segoe UI", 12, "bold"))
        self.canvas.create_text(211, 42 + lift, text="Z", fill="#baf8f6", font=("Segoe UI", 16, "bold"))


if __name__ == "__main__":
    BlackCatPet().run()
