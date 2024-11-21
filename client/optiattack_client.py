import uvicorn
from fastapi import FastAPI, File, UploadFile
from functools import wraps
import threading

from client import constants

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

def collect_info(host: str = constants.DEFAULT_CONTROLLER_HOST, port: int = constants.DEFAULT_CONTROLLER_PORT):
    """
    Decorator generator that:
    - Tracks method calls.
    - Automatically starts a FastAPI application.

    Args:
        host (str): Host where the FastAPI app will run.
        port (int): Port where the FastAPI app will run.
    """
    app = FastAPI()

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Increment the call count for the method
            return await func(*args, **kwargs)

        @app.get(f"/")
        async def test():
            return {"message": "Hello, FastAPI is running!"}

        # Register GET endpoint
        @app.get(f"{constants.INFO_NUT_PATH}")
        async def get_nut_state():
            return state

        # Register POST endpoint
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
        async def new_action(file: UploadFile = File(...)):
            return await func(file)

        # Start FastAPI server in a separate thread
        def start_server():
            uvicorn.run(app, host=host, port=port)

        threading.Thread(target=start_server, daemon=True).start()

        return wrapper

    return decorator
