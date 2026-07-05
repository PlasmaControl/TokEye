from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

from .load import (
    find_models,
    find_signals,
    is_model_cached,
    load_example_signal,
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


def refresh_dropdowns(signal_directory):
    models = list(MODEL_REGISTRY) + find_models()
    signals = find_signals(signal_directory)
    return [
        gr.Dropdown(choices=models),
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
    ]


def update_signal_dropdowns(signal_directory):
    """Update signal dropdowns when signal directory changes."""
    signals = find_signals(signal_directory)
    return [
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
        gr.Dropdown(choices=signals),
    ]


def toggle_view_groups(mode):
    return [
        gr.Group(visible=(mode == "Enhanced")),
        gr.Group(visible=(mode == "Mask" or mode == "Amplitude")),
    ]


def wrapper_model_load(model_file):
    """Load a model by registry name or local path.

    Registry names are downloaded from Hugging Face on first use (and cached
    thereafter); local paths are loaded directly. Failures are reported via
    a toast instead of raising, so a bad model choice degrades to "nothing
    loaded" rather than crashing the pipeline.
    """
    if not model_file:
        return None
    try:
        if model_file in MODEL_REGISTRY and not is_model_cached(model_file):
            gr.Info(f"Downloading {model_file} from Hugging Face (~30 MB, one-time)…")
        return model_load(model_file)
    except Exception as e:
        gr.Warning(f"Model load failed: {e}")
        return None


def ensure_model(model, model_file, signal_transform):
    """Load the model on first use; pass an already-loaded model through.

    Skips loading entirely (returning the model state unchanged) if no signal
    has been loaded yet: gr.Warning does not halt a .then() chain, so this
    gate is what prevents a pointless download/warmup on a no-signal click.
    """
    if signal_transform is None:
        gr.Warning("Load a signal first (or click Load Example Signal)")
        return model
    if model is not None:
        return model
    return wrapper_model_load(model_file)


def wrapper_load_single(signal_directory, signal_file, transform_args):
    """Wrapper to construct filepath from signal directory + signal file."""
    if not signal_directory or not signal_file or transform_args is None:
        return None
    return load_single(Path(signal_directory) / signal_file, transform_args)


def wrapper_load_multi(signal_directory, signal_1, signal_2, transform_args):
    """Wrapper to construct list of filepaths from signal directory + signal files."""
    if not signal_directory or not signal_1 or not signal_2 or transform_args is None:
        return None
    return load_multi(
        [Path(signal_directory) / signal_1, Path(signal_directory) / signal_2],
        transform_args,
    )


def wrapper_load_example(transform_args):
    """Wrapper to guard against missing transform state."""
    if transform_args is None:
        return None
    return load_example_signal(transform_args)


def analyze_tab():
    # User Interface
    with gr.Column():
        ## Refresh Page
        refresh_btn = gr.Button("Refresh Page")

        ## Model
        with gr.Group():
            model_file = gr.Dropdown(
                label="Analysis Model",
                info=(
                    "Built-in models download automatically from Hugging Face "
                    "on first load (~30 MB, cached). Local model/*.pt files "
                    "also listed."
                ),
                choices=list(MODEL_REGISTRY) + find_models(),
                value=DEFAULT_MODEL,
                interactive=True,
                allow_custom_value=True,
            )
            load_model_btn = gr.Button("Load Model")
        ## Transform
        with gr.Group():
            with gr.Group():
                clip_low_sld = gr.Slider(
                    0, 100, value=DEFAULT_CLIP_LOW, step=1, label="% Clip Low"
                )
                clip_high_sld = gr.Slider(
                    0, 100, value=DEFAULT_CLIP_HIGH, step=1, label="% Clip High"
                )
            with gr.Tab("STFT"):
                n_fft = gr.Slider(
                    256, 2048, value=DEFAULT_N_FFT, step=256, label="Number of Bins"
                )
                hop_length = gr.Slider(
                    64, 512, value=DEFAULT_HOP, step=64, label="Hop Size"
                )
                clip_dc = gr.Checkbox(
                    value=DEFAULT_CLIP_DC, label="Remove DC (Bottom) Bin"
                )
                setup_tranform_stft_btn = gr.Button("Apply Transform Settings")

        ## Signal Directory
        signal_directory = gr.Textbox(
            label="Signal Directory",
            value="data/input",
            info="Directory containing .npy signal files",
        )

        with gr.Tab("Single Signal Input"), gr.Column():
            signal_single = gr.Dropdown(
                label="Signal",
                info="Select Signal for Analysis",
                choices=[],
                interactive=True,
                allow_custom_value=True,
            )
            load_single_btn = gr.Button("Load Signal")
            load_example_btn = gr.Button("Load Example Signal")

        # Multi Signal
        with gr.Tab("Cross Signal Input"), gr.Column():
            signal_1 = gr.Dropdown(
                label="Signal 1",
                info="Select First Signal for Analysis",
                choices=[],
                interactive=True,
                allow_custom_value=True,
            )
            signal_2 = gr.Dropdown(
                label="Signal 2",
                info="Select Second Signal for Analysis",
                choices=[],
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
                    choices=["Original", "Enhanced", "Mask", "Amplitude"],
                    value="Enhanced",
                    label="View Mode",
                )
                with gr.Group():
                    out_1_chk = gr.Checkbox(
                        value=True,
                        label="Coherent Events",
                    )
                    out_2_chk = gr.Checkbox(
                        value=True,
                        label="Transient Events",
                    )
            with gr.Group(visible=True) as enhanced_grp, gr.Column():
                vmin_sld = gr.Slider(0, 100, value=0, step=1, label="% Min Clip")
                vmax_sld = gr.Slider(0, 100, value=100, step=1, label="% Max Clip")
            with gr.Group(visible=False) as mask_grp, gr.Column():
                threshold_sld = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")

            analyze_btn = gr.Button("Analyze", variant="primary")
            visualize_btn = gr.Button("Visualize")
            visualize_out = gr.Image(label="Visualization", type="pil")

    # State variables
    model = gr.State()
    signal_transform = gr.State()
    inference_output = gr.State()
    transform_args = gr.State(
        setup_stft_transform(
            DEFAULT_N_FFT,
            DEFAULT_HOP,
            DEFAULT_CLIP_DC,
            DEFAULT_CLIP_LOW,
            DEFAULT_CLIP_HIGH,
        )
    )

    # Event Handling
    ## Refresh Page
    refresh_btn.click(
        fn=refresh_dropdowns,
        inputs=[signal_directory],
        outputs=[model_file, signal_single, signal_1, signal_2],
    )

    ## Signal Directory - Update signal dropdowns when directory changes
    signal_directory.change(
        fn=update_signal_dropdowns,
        inputs=[signal_directory],
        outputs=[signal_single, signal_1, signal_2],
    )

    ## Model
    load_model_btn.click(
        fn=wrapper_model_load,
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
        fn=wrapper_load_single,
        inputs=[
            signal_directory,
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
        fn=wrapper_load_multi,
        inputs=[
            signal_directory,
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
    load_example_btn.click(
        fn=wrapper_load_example,
        inputs=[transform_args],
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

    ## One-click Analyze: ensure a model is loaded, run inference, visualize
    analyze_btn.click(
        fn=ensure_model,
        inputs=[model, model_file, signal_transform],
        outputs=[model],
    ).then(
        fn=model_infer,
        inputs=[signal_transform, model],
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
