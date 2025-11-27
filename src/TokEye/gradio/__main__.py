"""
TokEye Main Application

Time-series to spectrogram segmentation application for plasma physics analysis.
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
MAX_PORT_ATTEMPTS = 10
APP_TITLE = "TokEye"

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_theme() -> gr.Theme:
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    theme.set(
        button_primary_background_fill="*primary_300",
        button_primary_background_fill_hover="*primary_400",
        button_primary_text_color="white",
        button_secondary_background_fill="*secondary_200",
        button_secondary_background_fill_hover="*secondary_300",
        background_fill_primary="#f0f9ff",
        background_fill_secondary="#e0f2fe",
        border_color_primary="*primary_200",
        slider_color="*primary_400",
        input_background_fill="#ffffff",
        shadow_drop="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
        shadow_drop_lg="0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
    )
    return theme


def setup_css() -> str:
    css = """
    footer {display: none !important}
    .gradio-container {max-width: 100% !important}
    /* Additional beachy styling */
    .gradio-accordion {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 150, 150, 0.1) !important;
    }
    .gradio-button {
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .gradio-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 150, 150, 0.2) !important;
    }
    h1, h2, h3 {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    """
    return css


def check_logo() -> str | None:
    """
    Check if logo exists in assets/ directory.

    Returns:
        (logo_exists, logo_path)
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
        theme=setup_theme(),
        css=setup_css(),
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

        # Footer information
        gr.Markdown(
            """
            ---
            **TokEye** | nathaniel@princeton.edu
            """
        )

    return app


def parse_args():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Launch TokEye Application")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to run the server on"
    )
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument(
        "--open", action="store_true", help="Open browser automatically"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    cwd = Path.cwd()
    logging.info(f"TokEye initializing in: {cwd}")

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
