"""
TokEye Main Inference
"""

import argparse
import logging
from pathlib import Path

import gradio as gr

# Import tabs
from .tabs.analyze import analyze_tab
from .tabs.annotate import annotate_tab
from .tabs.utilities import utilities_tab

# Constants
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 5
APP_TITLE = "TokEye"

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def check_logo() -> str | None:
    """
    Check if logo exists in assets/ directory.

    Returns:
        logo_path: str | None
    """
    logo_path = Path.cwd() / "assets" / "logo.png"
    if logo_path.exists():
        return str(logo_path)
    return None


def create_app() -> gr.Blocks:
    """Create the TokEye Gradio application."""
    logo_path = check_logo()
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Ocean(),
    ) as app:
        with gr.Row():
            with gr.Column(scale=1):
                ...
            with gr.Column(scale=2):
                gr.Image(
                    logo_path,
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=150,
                    show_download_button=False,
                    show_share_button=False,
                )
                gr.Markdown(
                    """
                    # TokEye
                    """,
                    elem_classes="center-text",
                )
            with gr.Column(scale=1):
                pass  # Empty column for centering

        # Tabs
        with gr.Tabs():
            with gr.Tab("Analyze"):
                analyze_tab()
            with gr.Tab("Annotate"):
                annotate_tab()
            with gr.Tab("Utilities"):
                utilities_tab()

    return app


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description="Launch TokEye")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--share", action="store_true", help="Public link")
    parser.add_argument("--open", action="store_true", help="Open browser")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    cwd = Path.cwd()
    logging.info(f"Initializing TokEye in: {cwd}")

    for directory in ["cache", "outputs", "annotations", "model", "data"]:
        (cwd / directory).mkdir(exist_ok=True)

    app = create_app()

    current_port = args.port
    for attempt in range(MAX_PORT_ATTEMPTS):
        try:
            logging.info(f"Launching on port {current_port}...")
            app.launch(
                server_port=current_port,
                share=args.share,
                inbrowser=args.open
                if attempt == 0
                else False,  # Only open browser on first attempt
                show_error=True,
            )
            break
        except OSError:
            logging.warning(f"Port {current_port} is in use.")
            current_port += 1
            if attempt == MAX_PORT_ATTEMPTS - 1:
                logging.error("Could not find an open port.")
                raise


if __name__ == "__main__":
    main()
