from __future__ import annotations

import shutil
import sys
import time

term_width = shutil.get_terminal_size((80, 20)).columns

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current: int, total: int, msg: str | None = None) -> None:
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    sys.stdout.write("=" * cur_len)
    sys.stdout.write(">")
    sys.stdout.write("." * rest_len)
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    parts: list[str] = []
    parts.append(f"  Step: {_format_time(step_time)}")
    parts.append(f" | Tot: {_format_time(tot_time)}")
    if msg:
        parts.append(" | " + msg)
    full_msg = "".join(parts)

    sys.stdout.write(full_msg)
    pad = term_width - int(TOTAL_BAR_LENGTH) - len(full_msg) - 3
    if pad > 0:
        sys.stdout.write(" " * pad)

    # Go back to the center of the bar.
    back = term_width - int(TOTAL_BAR_LENGTH / 2) + 2
    if back > 0:
        sys.stdout.write("\b" * back)
    sys.stdout.write(f" {current + 1}/{total} ")

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _format_time(seconds: float) -> str:
    days = int(seconds / 3600 / 24)
    seconds -= days * 3600 * 24
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    secondsf = int(seconds)
    seconds -= secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    return f or "0ms"

