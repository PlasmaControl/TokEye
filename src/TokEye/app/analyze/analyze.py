import logging

import gradio as gr

from .load import (
    get_available_models,
    get_available_signals,
    load_multi,
    load_single,
    model_infer,
    model_load,
)
from .visualize import (
    show_image,
)

logger = logging.getLogger(__name__)


def setup_stft_transform(n_fft, hop_length, clip_dc, clip_low, clip_high):
    """Create transform args dict for state storage."""
    return {
        "type": "stft",
        "n_fft": n_fft,
        "hop_length": hop_length,
        "clip_dc": clip_dc,
        "percentile_low": clip_low,
        "percentile_high": clip_high,
    }


def refresh_dropdowns():
    """Refresh model and signal dropdowns."""
    models = get_available_models()
    signals = get_available_signals()
    return [
        gr.Dropdown(choices=models),
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
    ]


def toggle_view_groups(mode):
    """Toggle visibility of enhanced and mask groups based on view mode."""
    return [
        gr.Group(visible=(mode == "Enhanced")),
        gr.Group(visible=(mode == "Mask")),
    ]


def analyze_tab():
    # User Interface
    with gr.Column():
        ## Refresh Page
        refresh_btn = gr.Button("Refresh Page")

        ## Model
        with gr.Group():
            model_file = gr.Dropdown(
                label="Analysis Model",
                info="Select Model For Analysis",
                choices=get_available_models(),
                interactive=True,
                allow_custom_value=True,
            )
            load_model_btn = gr.Button("Load Model")
        ## Transform
        with gr.Group():
            with gr.Group():
                clip_low_sld = gr.Slider(0, 100, value=1, step=1, label="% Clip Low")
                clip_high_sld = gr.Slider(0, 100, value=99, step=1, label="% Clip High")
            with gr.Tab("STFT"):
                n_fft = gr.Slider(
                    256, 2048, value=1024, step=256, label="Number of Bins"
                )
                hop_length = gr.Slider(64, 512, value=256, step=64, label="Hop Size")
                clip_dc = gr.Checkbox(value=True, label="Remove DC (Bottom) Bin")
                setup_tranform_stft_btn = gr.Button("Setup Transform")

        with gr.Tab("Single Signal Input"), gr.Column():
            signal_single = gr.Dropdown(
                label="Signal",
                info="Select Signal for Analysis",
                choices=get_available_signals(),
                interactive=True,
                allow_custom_value=True,
            )
            load_single_btn = gr.Button("Load Signal")

        # Multi Signal
        with gr.Tab("Cross Signal Input"), gr.Column():
            signal_1 = gr.Dropdown(
                label="Signal 1",
                info="Select First Signal for Analysis",
                choices=get_available_signals(),
                interactive=True,
                allow_custom_value=True,
            )
            signal_2 = gr.Dropdown(
                label="Signal 2",
                info="Select Second Signal for Analysis",
                choices=get_available_signals(),
                interactive=True,
                allow_custom_value=True,
            )
            load_multi_btn = gr.Button("Load Signal")

        # Extraction Visualization
        extract_out = gr.Image(
            label="Extraction Output",
            type="pil",
        )

        # Visualization
        with gr.Column():
            with gr.Row():
                view_mode = gr.Radio(
                    choices=["Original", "Enhanced", "Mask"],
                    value="Enhanced",
                    label="View Mode",
                )
                with gr.Group():
                    out_1_chk = gr.Checkbox(
                        value=False,
                        label="Coherent Events",
                    )
                    out_2_chk = gr.Checkbox(
                        value=False,
                        label="Transient Events",
                    )
            with gr.Group(visible=True) as enhanced_grp, gr.Column():
                vmin_sld = gr.Slider(0, 100, value=1, step=1, label="% Min Clip")
                vmax_sld = gr.Slider(0, 100, value=99, step=1, label="% Max Clip")
            with gr.Group(visible=False) as mask_grp, gr.Column():
                threshold_sld = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")

            visualize_btn = gr.Button("Visualize")
            visualize_out = gr.Image(label="Visualization", type="pil")

    # State variables
    model = gr.State()
    signal_transform = gr.State()
    inference_output = gr.State()
    transform_args = gr.State()

    # Event Handling
    ## Refresh Page
    refresh_btn.click(
        fn=refresh_dropdowns,
        inputs=[],
        outputs=[model_file, signal_single, signal_1, signal_2],
    )

    ## Model
    load_model_btn.click(
        fn=model_load,
        inputs=[model_file],
        outputs=[model],
    )

    ## Transform
    setup_tranform_stft_btn.click(
        fn=setup_stft_transform,
        inputs=[
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
        ],
        outputs=[transform_args],
    )

    ## Signal
    load_single_btn.click(
        fn=load_single,
        inputs=[
            signal_single,
            transform_args,
        ],
        outputs=[signal_transform],
    ).then(
        fn=show_image,
        inputs=[
            gr.State("Original"),
            signal_transform,
            inference_output,
            gr.State(False),
            gr.State(False),
            vmin_sld,
            vmax_sld,
            threshold_sld,
        ],
        outputs=[extract_out],
    )
    load_multi_btn.click(
        fn=load_multi,
        inputs=[
            signal_1,
            signal_2,
            transform_args,
        ],
        outputs=[signal_transform],
    ).then(
        fn=show_image,
        inputs=[
            gr.State("Original"),
            signal_transform,
            inference_output,
            gr.State(False),
            gr.State(False),
            vmin_sld,
            vmax_sld,
            threshold_sld,
        ],
        outputs=[extract_out],
    )

    ## Visualization
    view_mode.change(
        fn=toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp],
    )

    visualize_btn.click(
        fn=model_infer,
        inputs=[
            signal_transform,
            model,
        ],
        outputs=[inference_output],
    ).then(
        fn=show_image,
        inputs=[
            view_mode,
            signal_transform,
            inference_output,
            out_1_chk,
            out_2_chk,
            vmin_sld,
            vmax_sld,
            threshold_sld,
        ],
        outputs=[visualize_out],
    )
