"""Individual search functions."""


class Individual:

    """Individual search functions."""

    def __init__(self):
        """Initializes an individual with an empty list of actions."""
        self.actions = []
        self.individual_origin = None

    def add_action(self, action):
        """Add an action to the individual."""
        does_contain = [action.same_location(a) for a in self.actions]
        if any(does_contain):
            return False

        if action not in self.actions:
            self.actions.append(action)
            return True

    def get_actions(self):
        """Get the actions of the individual."""
        return self.actions

    def set_individual_origin(self, individual_origin):
        """Set the individual origin."""
        self.individual_origin = individual_origin

    def size(self):
        """Get the number of actions in the individual."""
        return len(self.actions)

    def copy(self):
        """Copy the individual."""
        new_individual = Individual()
        for action in self.actions:
            new_individual.add_action(action.copy())
        return new_individual

    def get_action_image(self, action_image):
        """Get the image."""
        for action in self.actions:
            x, y = action.get_location()
            action_image[x, y] = action.get_color()
        return action_image
