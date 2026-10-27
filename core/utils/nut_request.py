"""NutRequest class for handling requests to the Nut API."""

from typing import List, Optional

from core.utils.label import Label


class NutRequest:

    """NutRequest class for handling requests to the Nut API."""

    def __init__(self, request, target=None):
        """Initializes a NutRequest object with the provided request."""
        predictions = request["predictions"]
        self.predictions = NutRequest.request_to_label_array(predictions)
        self.max_score = max(self.predictions, key=lambda x: x.value)
        self.second_max_score = sorted(self.predictions, key=lambda x: x.value, reverse=True)[1]

        # Optional logits of all classes in the NUT's label order (white-box NUTs); None when not reported.
        raw_logits = request.get("logits")
        self.logits: Optional[List[float]] = [float(v) for v in raw_logits] if raw_logits is not None else None

        self.target_score = next(
            (label for label in self.predictions if label.label == target), Label(target, 0.0)
        ) if target else None

    @staticmethod
    def request_to_label(prediction):
        """Converts a prediction dictionary to a Label object."""

        return Label(prediction["label"], prediction["score"])

    @staticmethod
    def request_to_label_array(predictions):
        """Converts a list of prediction dictionaries to a list of Label objects."""

        return [NutRequest.request_to_label(prediction) for prediction in predictions]
