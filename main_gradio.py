import gradio as gr
from main import OptiAttack
from core.problem.base_module import BaseModule
from PIL import  Image
import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


sys.stdout = Logger("output.log")

def run_optiattack(host_address, port_number, input_image_path, image_width,
                               image_height, max_evals, mutation_sigma):
    container = BaseModule()
    config_parser = container.config_parser()
    if not input_image_path is None:
        image = Image.fromarray(input_image_path)
        image.save("./tests/test_img_custom.jpeg")
    parsed_args = config_parser.parse_args()
    parsed_args["image_width"] = image_width
    parsed_args["image_height"] = image_height
    parsed_args["host_address"] = host_address
    parsed_args["port_number"] = port_number
    parsed_args["input_image_path"] = "./tests/test_img_custom.jpeg"
    parsed_args["max_evals"] = max_evals
    parsed_args["mutation_sigma"] = mutation_sigma
    container.config.override(parsed_args)

    app = OptiAttack()
    container.wire(modules=[app])
    app.startup()
    image = app.run()
    return image

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            host_address = gr.Text(value="localhost", label="Host Address:")
            port_number = gr.Text(value="8080", label="Port Number")
        input_image_path = gr.Image(label="Input Image")
        with gr.Row():
            image_width = gr.Number(value=224, label="Image Width")
            image_height = gr.Number(value=224, label="Image Height")
        with gr.Row():
            max_evals = gr.Number(value=1000, label="Max Evaluations")
            mutation_sigma = gr.Number(value=50, label="Mutation Sigma")
        btn = gr.Button(value="Run OptiAttack")
        map = gr.Image()
        logs = gr.Textbox(label="Console Output")
    btn.click(run_optiattack, [host_address, port_number, input_image_path, image_width,
                               image_height, max_evals, mutation_sigma], map)
    t = gr.Timer(1, active=True)
    t.tick(read_logs, outputs=logs)


if __name__ == "__main__":
    demo.launch()