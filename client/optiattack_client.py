import numpy as np
import uvicorn

from fastapi import FastAPI
from functools import wraps
import threading

from pydantic import BaseModel

try:
    from client import constants
except ImportError:
    import constants

# State dictionary to track method call counts
state = {
    "is_running": False,
    "controller_host": "",
    "controller_port": None,
    "run_nut": constants.RUN_NUT_PATH,
    "stop_nut": constants.STOP_NUT_PATH,
    "info_nut": constants.INFO_NUT_PATH,
    "new_action": constants.NEW_ACTION,
}
app = FastAPI()


class MatrixModel(BaseModel):
    image: list[list[list[int]]]


def collect_info(host: str = constants.DEFAULT_CONTROLLER_HOST,
                 port: int = constants.DEFAULT_CONTROLLER_PORT):
    """
    Decorator generator that:
    - Tracks method calls.
    - Automatically starts a FastAPI application.

    Args:
        host (str): Host where the FastAPI app will run.
        port (int): Port where the FastAPI app will run.
    """

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @app.get("/")
        async def test():
            return {"message": "Hello, FastAPI is running!"}

        @app.get(f"{constants.INFO_NUT_PATH}")
        async def get_nut_state():
            return state

        @app.post(f"{constants.RUN_NUT_PATH}")
        async def run_nut():
            state["is_running"] = True
            state["controller_host"] = host
            state["controller_port"] = port

            return state

        @app.post(f"{constants.STOP_NUT_PATH}")
        async def stop_nut():
            state["is_running"] = False
            state["controller_host"] = ""
            state["controller_port"] = None
            return state

        @app.post(f"{constants.NEW_ACTION}")
        async def new_action(data: MatrixModel):
            array_data = np.array(data.image)

            return func(array_data)

        def start_server():
            uvicorn.run(app, host=host, port=port)

        wrapper.app = app
        threading.Thread(target=start_server, daemon=True).start()

        return wrapper

    return decorator
