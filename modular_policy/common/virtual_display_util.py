import sys
import os
from typing import Tuple


def start_virtual_display(size: Tuple[int, int] = (1280, 1024)):
    # starts Xvfb
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        try:
            from pyvirtualdisplay import Display
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Timeout starting virtual display")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            virtual_display = Display(visible=0, size=size)
            virtual_display.start()
            signal.alarm(0)
        except Exception as e:
            print(f"Failed to start virtual display: {e}")