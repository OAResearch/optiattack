# Singleton decorator
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


# Custom decorator to add descriptions to parameters
def Cfg(description: str):
    def decorator(func):
        func.cfg_description = description
        return func

    return decorator
