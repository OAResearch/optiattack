"""Custom decorator to add descriptions to configuration parameters."""


def Cfg(description: str):
    """Decorate to add descriptions to configuration parameters."""

    def decorator(func):
        func.cfg_description = description
        return func

    return decorator
