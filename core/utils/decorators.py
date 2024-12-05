# Custom decorator to add descriptions to parameters
def Cfg(description: str):
    def decorator(func):
        func.cfg_description = description
        return func

    return decorator
