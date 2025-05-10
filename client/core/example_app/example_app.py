"""Example app to test the client."""

from client.core.optiattack_client import collect_info


@collect_info()
async def example_method(image):
    """Example method to test the client."""
    return {"message": f"file {image} reached successfully!"}

while True:
    pass
