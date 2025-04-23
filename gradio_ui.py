import os

import gradio as gr

from core.utils.application import configure_container
from main import OptiAttack
from core.problem.base_module import BaseModule
from PIL import  Image
import sys


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        m = strip_ascii(message)
        self.terminal.write(m)
        self.log.write(m)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

if not os.path.exists("./.cache"):
    os.makedirs("./.cache")

sys.stdout = Logger("./.cache/output.log")

def strip_ascii(text):
    res = "".join(
        char for char
        in text
        if not ord(char) == 27
    )
    res = res.replace("[1A", "")
    res = res.replace("[2K", "")
    return res

def read_logs():
    sys.stdout.flush()
    with open("./.cache/output.log", "r") as f:
        return f.read()

def clear_log():
    open("./.cache/output.log", "w")

def delete_cache():
    import shutil
    shutil.rmtree("./.cache", ignore_errors=True)

def run_optiattack(host_address, port_number, input_image_path, image_width,
                   image_height, max_evals, ):

    container = BaseModule()
    config_parser = container.config_parser()
    if not input_image_path is None:
        image = Image.fromarray(input_image_path)
        image.save("./.cache/cache_img.jpeg")
    parsed_args = config_parser.parse_args()
    parsed_args["image_width"] = image_width
    parsed_args["image_height"] = image_height
    parsed_args["nut_host"] = host_address
    parsed_args["nut_port"] = port_number
    parsed_args["input_image"] = "./.cache/cache_img.jpeg"
    parsed_args["max_evaluations"] = max_evals
    container.config.override(parsed_args)
    container = configure_container(container)

    app = OptiAttack()
    container.wire(modules=[app])
    app.startup()
    folders = app.run()
    image_folder = folders["images_folder"]
    final_image_path = f"{image_folder}/final_image.jpg"
    line_image_path = f"{image_folder}/line.png"
    matrix_overlay_path = f"{image_folder}/matrix_overlay.png"

    final_image = Image.open(final_image_path)
    line_image = Image.open(line_image_path)
    matrix_overlay_image = Image.open(matrix_overlay_path)

    statistics_folder = folders["statistics_folder"]
    statistics_file_path = os.path.join(statistics_folder + '/data.json')
    if os.path.exists(statistics_file_path):
        statistics_file = open(statistics_file_path)
        report_text = statistics_file.read()
    else:
        report_text = ""

    return final_image, line_image, matrix_overlay_image, report_text

with gr.Blocks() as web_app:
    with gr.Column():
        title = gr.HTML('</br><h1 style="text-align:center">OptiAttack: Optimization-Based Adversarial Example Generation Software</h1></br>')
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<h2 style="text-align:center">Parameters</h2>')
                with gr.Row():
                    host_address = gr.Text(value="localhost", label="Host Address:")
                    port_number = gr.Text(value="38000", label="Port Number")
                with gr.Row():
                    image_width = gr.Number(value=224, label="Image Width")
                    image_height = gr.Number(value=224, label="Image Height")
                with gr.Row():
                    max_evals = gr.Number(value=1000, label="Max Evaluations")
            with gr.Column(scale=2, min_width=300):
                gr.HTML('<h2 style="text-align:center">Input Image</h2>')
                input_image_path = gr.Image(label="Input Image", height=300, width=300)
        with gr.Row():
            btn_clr = gr.Button(value="Clear", size="lg")
            btn_run = gr.Button(value="Run OptiAttack", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<h2 style="text-align:center">Final Image</h2>')
                final_image = gr.Image(label="Final Image")
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<h2 style="text-align:center">Confidence Score Chart</h2>')
                line_image = gr.Image(label="Line Image")
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<h2 style="text-align:center">Adversarial Attack Image</h2>')
                matrix_overlay_image = gr.Image(label="Matrix Overlay Image")

        with gr.Tab("Console Output"):
            logs = gr.Textbox(label="Console Output", lines=15)
        with gr.Tab("Report"):
            report_text = gr.Textbox(label="Report", lines=15)

    def clear_input_outputs():
        clear_log()
        return {
            host_address: "localhost",
            port_number: 38000,
            image_width: 224,
            image_height: 224,
            max_evals: 1000,
            input_image_path:  None,
            final_image:  None,
            line_image: None,
            matrix_overlay_image: None,
            btn_clr: gr.update(interactive=True),
            btn_run: gr.update(interactive=True),
            report_text: "",
            logs: ""
        }


    btn_run.click(fn=run_optiattack, inputs=[host_address, port_number, input_image_path, image_width,
                               image_height, max_evals],
                  outputs=[final_image, line_image, matrix_overlay_image, report_text])
    btn_clr.click(fn=clear_input_outputs, inputs=[], outputs=[host_address, port_number, image_width, image_height,
                                                              max_evals, input_image_path, final_image, line_image,
                                                              matrix_overlay_image, btn_clr, btn_run, report_text, logs])
    t = gr.Timer(1, active=True)
    t.tick(read_logs, outputs=logs)
    web_app.unload(delete_cache)