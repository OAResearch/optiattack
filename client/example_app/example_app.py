from client.optiattack_client import collect_info


@collect_info()
async def example_method(image):
    return {"message": f"file {image} reached successfully!"}

while True:
    pass
