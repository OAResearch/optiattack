from client.optiattack_client import collect_info


@collect_info()
async def example_method(file):
    return {"message": f"file {file.filename} reached successfully!"}

while True:
    pass
