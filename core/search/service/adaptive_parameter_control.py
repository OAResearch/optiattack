"""Adaptive Parameter Control module."""

from core.search.service.search_time_controller import SearchTimeController


class AdaptiveParameterControl:

    """Class for the adaptive parameter control."""

    def __init__(self, stc: SearchTimeController, config: dict):
        """Initialize the adaptive parameter control."""
        self.stc = stc
        self.config = config

    def get_dpc_value(self, start, end, start_time, threshold):
        """Get the dynamic parameter control value."""
        passed = self.stc.percentage_used_budget()

        if passed < start_time:
            return start

        if passed >= threshold:
            return end

        scale = (passed - start_time) / (threshold - start_time)
        delta = end - start

        return start + delta * scale

    def get_pixel_apc(self):
        """Get the pixel value for the adaptive parameter control."""

        return self.get_dpc_value(self.config.get("apc_pixel_start"), self.config.get("apc_pixel_end"),
                                  self.config.get("apc_start_time"), self.config.get("apc_threshold"))

    def get_location_apc(self):
        """Get the location value for the adaptive parameter control."""

        return self.get_dpc_value(self.config.get("apc_location_start"), self.config.get("apc_location_end"),
                                  self.config.get("apc_start_time"), self.config.get("apc_threshold"))

    def get_probability_random_sampling(self):
        """Get the probability of sampling a new individual at random."""

        return self.get_dpc_value(self.config.get("random_sampling_probability"), 0,
                                  self.config.get("apc_start_time"), self.config.get("focused_search_activation_time"))
