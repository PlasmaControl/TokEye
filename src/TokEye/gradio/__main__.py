"""
TokEye Main Application

Time-series to spectrogram segmentation application for plasma physics analysis.
"""

import gradio as gr
import sys
from pathlib import Path
import logging

# Constants
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tabs
from .tabs.analyze import analyze_tab
from .tabs.annotate import annotate_tab
from .tabs.utilities import utilities_tab


def check_logo() -> tuple[bool, str]:
    """
    Check if logo exists in assets/ directory.

    Returns:
        (logo_exists, logo_path)
    """
    logo_path = project_root / "assets" / "logo.png"
    if logo_path.exists():
        return True, str(logo_path)
    return False, ""


def create_app() -> gr.Blocks:
    """Create the TokEye Gradio application."""

    # Check for logo
    has_logo, logo_path = check_logo()

    # Create main application
    with gr.Blocks(
        title="TokEye - Plasma Signal Segmentation",
        theme=gr.themes.Soft(),
        css="""
        footer {display: none !important}
        .gradio-container {max-width: 100% !important}
        """
    ) as app:

        # Title screen with logo if available
        if has_logo:
            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Empty column for centering
                with gr.Column(scale=2):
                    gr.Image(
                        logo_path,
                        label=None,
                        show_label=False,
                        interactive=False,
                        container=False,
                        height=300,  # Larger
                        show_download_button=False,
                        show_share_button=False
                    )
                    gr.Markdown(
                        """
                        # TokEye - Plasma Signal Segmentation
                        **Advanced time-series analysis for tokamak plasma diagnostics**
                        """,
                        elem_classes="center-text"
                    )
                with gr.Column(scale=1):
                    pass  # Empty column for centering
        else:
            gr.Markdown("# TokEye - Plasma Signal Segmentation")
            gr.Markdown("**Advanced time-series analysis for tokamak plasma diagnostics**")

        # Tabs
        with gr.Tabs():
            with gr.Tab("🔬 Analyze"):
                analyze_tab()

            with gr.Tab("✏️ Annotate"):
                annotate_tab()

            with gr.Tab("🔧 Utilities"):
                utilities_tab()

        # Footer information
        gr.Markdown(
            """
            ---
            **TokEye** | Plasma Control Group | Time-series to Spectrogram Segmentation
            """
        )

    return app


def get_port_from_args() -> int:
    """Parse port from command line arguments."""
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            try:
                return int(sys.argv[port_index])
            except ValueError:
                logging.warning(f"Invalid port specified, using default {DEFAULT_PORT}")
    return DEFAULT_PORT


def launch_gradio(app: gr.Blocks, port: int):
    """Launch the Gradio application."""
    share = "--share" in sys.argv
    open_browser = "--open" in sys.argv

    logging.info(f"Launching TokEye on port {port}")
    if share:
        logging.info("Share mode enabled - creating public link")

    app.launch(
        server_port=port,
        share=share,
        inbrowser=open_browser,
        show_error=True,
    )


def main():
    """Main entry point for TokEye application."""

    # Create necessary directories
    directories = ["cache", "outputs", "annotations", "model", "data"]
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)

    logging.info("TokEye initializing...")
    logging.info(f"Project root: {project_root}")

    # Create application
    app = create_app()

    # Get port from arguments
    port = get_port_from_args()

    # Try to launch with port fallback
    for attempt in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(app, port)
            break
        except OSError as e:
            if attempt < MAX_PORT_ATTEMPTS - 1:
                logging.warning(
                    f"Port {port} unavailable, trying {port + 1}..."
                )
                port += 1
            else:
                logging.error(f"Failed to launch after {MAX_PORT_ATTEMPTS} attempts")
                raise
        except Exception as error:
            logging.error(f"An error occurred launching Gradio: {error}")
            raise


if __name__ == "__main__":
    main()
