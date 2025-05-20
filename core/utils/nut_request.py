"""NutRequest class for handling requests to the Nut API."""

from core.utils.label import Label


class NutRequest:

    """NutRequest class for handling requests to the Nut API."""

    def __init__(self, request, target=None):
        """Initializes a NutRequest object with the provided request."""
        predictions = request["predictions"]
        self.predictions = NutRequest.request_to_label_array(predictions)
        self.max_score = max(self.predictions, key=lambda x: x.value)
        self.second_max_score = sorted(self.predictions, key=lambda x: x.value, reverse=True)[1]
        # Get the score for the target class (if it exists in predictions)
        if target:
            self.targeted_score = next((label for label in self.predictions if label.label == target),
                                       Label(target, 0.0))  # Set to 0.0 if target not found
            print(self.targeted_score)
        else:
            self.targeted_score = Label("None", 0.0)

    @staticmethod
    def request_to_label(prediction):
        """Converts a prediction dictionary to a Label object."""

        return Label(prediction["label"], prediction["score"])

    @staticmethod
    def request_to_label_array(predictions):
        """Converts a list of prediction dictionaries to a list of Label objects."""

        return [NutRequest.request_to_label(prediction) for prediction in predictions]
