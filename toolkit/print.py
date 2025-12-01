import sys
import os
from toolkit.accelerator import get_accelerator


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        try:
            # Use unbuffered mode (buffering=1 means line buffered, but we want immediate writes)
            # Open in append mode with line buffering
            self.log = open(filename, 'a', buffering=1)
            # Force a write to ensure file is created and writable
            self.log.write("")
            self.log.flush()
        except Exception as e:
            # If we can't open the log file, at least try to write to terminal
            print(f"Warning: Could not open log file {filename}: {e}", file=sys.stderr)
            self.log = None

    def write(self, message):
        try:
            if self.log is not None:
                self.log.write(message)
                self.log.flush()  # Make sure it's written immediately
        except Exception:
            # If log write fails, continue silently to avoid breaking the process
            pass
        try:
            self.terminal.write(message)
        except (OSError, AttributeError):
            # Terminal might be closed/ignored when spawned with stdio: 'ignore'
            pass

    def flush(self):
        try:
            if self.log is not None:
                self.log.flush()
        except Exception:
            pass
        try:
            self.terminal.flush()
        except (OSError, AttributeError):
            pass


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    sys.stdout = Logger(filename)
    sys.stderr = Logger(filename)
