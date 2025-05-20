"""NutRequest class for handling requests to the Nut API."""

from core.utils.label import Label


class NutRequest:

    """NutRequest class for handling requests to the Nut API."""

    def __init__(self, request):
        """Initializes a NutRequest object with the provided request."""
        predictions = request["predictions"]
        self.predictions = NutRequest.request_to_label_array(predictions)
        self.max_score = max(self.predictions, key=lambda x: x.value)
        self.second_max_score = sorted(self.predictions, key=lambda x: x.value, reverse=True)[1]


    @staticmethod
    def request_to_label(prediction):
        """Converts a prediction dictionary to a Label object."""

        return Label(prediction["label"], prediction["score"])

    @staticmethod
    def request_to_label_array(predictions):
        """Converts a list of prediction dictionaries to a list of Label objects."""

        return [NutRequest.request_to_label(prediction) for prediction in predictions]
