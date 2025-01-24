"""Module contains the Label class."""


class Label:

    """Label class for handling labels."""

    def __init__(self, label, value):
        """Initializes a Label object with the provided label and value"""

        self.label = label
        self.value = value

    def __str__(self):
        """Returns a string representation of the Label object."""

        return f"{self.label}: {self.value}"
