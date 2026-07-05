"""
TokEye Main Inference
"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
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

logger = logging.getLogger(__name__)


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=make_theme(),
        css="footer{display:none !important}",
    ) as app:
        logo_path = importlib.resources.files("tokeye.app").joinpath("assets/logo.png")
        if logo_path.is_file():
            gr.Image(
                str(logo_path),
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


def main(
    port: int = DEFAULT_PORT,
    share: bool = False,
    open_browser: bool = False,
) -> None:
    logger.info(f"Initializing TokEye in: {Path.cwd()}")
    app = create_app()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            app.launch(
                share=share,
                inbrowser=open_browser,
                server_port=port,
            )
        except OSError:
            print(f"Failed on port {port}")
            port -= 1
        except Exception as error:
            print(f"{error}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m tokeye.app",
        description="Launch the TokEye Gradio app.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to serve the app on."
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the app in a browser on launch.",
    )
    args = parser.parse_args()
    main(port=args.port, share=args.share, open_browser=args.open_browser)
