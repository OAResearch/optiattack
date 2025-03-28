"""Individual search functions."""


class Individual:

    """Individual search functions."""

    def __init__(self):
        """Initializes an individual with an empty list of actions."""
        self.actions = []
        self.individual_origin = None

    def add_action(self, action, replace=True):
        """Add an action to the individual."""
        does_contain = [action.same_location(a) for a in self.actions]

        if not replace and any(does_contain):
            # If the action already exists in the individual and we don't want to replace it, return False
            return False

        if any(does_contain):
            # If the action already exists in the individual, replace it
            index = does_contain.index(True)
            self.actions[index] = action
            return True

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

    def __eq__(self, other):
        """Check if two individuals are equal."""
        if not isinstance(other, Individual):
            return False
        return self.actions == other.actions and self.individual_origin == other.individual_origin
