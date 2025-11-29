"""
TokEye Main Inference
"""

import logging
import sys
from pathlib import Path

import gradio as gr

# Import tabs
from .analyze.analyze import analyze_tab
from .tabs.annotate import annotate_tab
from .tabs.utilities import utilities_tab
from .utils.theme import make_theme

# Constants
APP_TITLE = "TokEye"
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Current working directory
cwd = Path.cwd()


def create_directories():
    for directory in ["cache", "outputs", "annotations", "model", "data"]:
        (cwd / directory).mkdir(exist_ok=True)


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=make_theme(),
        css="footer{display:none !important}",
    ) as app:
        gr.Image(
            str(Path.cwd() / "assets" / "logo.png"),
            height=300,
            interactive=False,
            container=False,
            show_download_button=False,
            show_fullscreen_button=False,
        )
        with gr.Tab("Analyze"):
            analyze_tab()
        with gr.Tab("Annotate"):
            annotate_tab()
        with gr.Tab("Utilities"):
            utilities_tab()
    return app


def get_port():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


def launch(app, port):
    app.launch(
        # favicon_path="assets/ICON.ico", # TODO: Set up favicon
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
    )


if __name__ == "__main__":
    logging.info(f"Initializing TokEye in: {cwd}")
    # Set up
    create_directories()
    app = create_app()
    # Launch application
    port = get_port()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch(app, port)
        except OSError:
            print(f"Failed on port {port}")
            port -= 1
        except Exception as error:
            print(f"{error}")
            break
