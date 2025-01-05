"""Module used to compute an average of values incrementally, each time a new value is provided."""

import time


class IncrementalAverage:

    """Class used to compute an average of values incrementally, each time a new value is provided."""

    def __init__(self):
        """Initialize the incremental average."""
        # The number [n] of total values
        self.n = 0

        # The average value. For simplicity, considering 0 if no elements (n == 0)
        self.mean = 0.0

        # Min and max values
        self.min = 0.0
        self.max = 0.0

        # Timer-related variables
        self.starting_time = None

    def add_value(self, k):
        """Add a new value to compute the incremental average. This will be converted into a float."""
        d = float(k)

        if self.n == 0:
            self.min = d
            self.max = d
        else:
            if d < self.min:
                self.min = d
            if d > self.max:
                self.max = d

        self.n += 1

        self.mean = self.mean + ((d - self.mean) / self.n)

    def start_timer(self):
        """Start the timer."""
        self.starting_time = time.time() * 1000  # Convert to milliseconds

    def is_recording_timer(self):
        """Check if the timer is currently running."""
        return self.starting_time is not None

    def add_elapsed_time(self):
        """Add the number of milliseconds since the timer was started, and then reset it."""
        if self.starting_time is None:
            raise RuntimeError("Adding elapsed time before starting the timer")

        elapsed = (time.time() * 1000) - self.starting_time  # Calculate elapsed time in milliseconds
        self.add_value(elapsed)
        self.starting_time = None  # Reset the timer
        return elapsed

    def __str__(self):
        """Return a string representation of the average, min, and max values."""
        return f"Avg={self.mean:.2f}, min={self.min:.2f}, max={self.max:.2f}"
