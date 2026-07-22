from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from tokeye import export
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

GETTING_STARTED_MD = """\
**1. Save your signal as a 1D float NumPy array** (raw time series — any \
length, no normalization or preprocessing needed):

```python
import numpy as np

np.save("data/input/myshot.npy", signal)  # signal: 1D float array
```

**2. Point *Signal Directory* at that folder** — the signal dropdowns update \
automatically (or click *Refresh Page*).

**3. Load the signal, then click *Analyze*.** The built-in `big_tf_unet` \
model downloads automatically on first use (~30 MB, cached).

No data yet? Click **Load Example Signal** below, or run `tokeye example` in \
a terminal. Audio files (.wav/.mp3/...)? Convert them to `.npy` in the \
**Utilities** tab.
"""


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


def wrapper_model_load_pair(model_file):
    """Load a model and report which name is now loaded.

    Wraps ``wrapper_model_load`` so ``load_model_btn.click`` can populate both
    the ``model`` and ``model_name`` states in one step — ``model_name`` is
    what lets ``ensure_model`` later detect a dropdown change without a
    reload. ``model_name`` is ``None`` when the load failed (mirrors
    ``wrapper_model_load``'s "nothing loaded" degradation, reported via a
    toast rather than an exception).
    """
    model = wrapper_model_load(model_file)
    return model, (model_file if model is not None else None)


def ensure_model(
    model,
    loaded_name,
    model_file,
    signal_transform,
    progress=gr.Progress(),
):
    """Load the model if it's missing or stale; pass a fresh one through.

    (Re)loads via ``wrapper_model_load`` whenever ``model is None`` (nothing
    loaded yet, or a previous load failed) or ``loaded_name != model_file``
    (the dropdown has changed since the last load) — this is what makes
    switching models in the dropdown take effect on the next Analyze click,
    instead of being silently ignored until a separate "Load Model" click.

    Skips loading entirely (returning the state unchanged) if no signal has
    been loaded yet: gr.Warning does not halt a .then() chain, so this gate
    is what prevents a pointless download/warmup on a no-signal click.

    ``progress`` must stay a trailing PLAIN (positional-or-keyword) default
    parameter, NOT keyword-only: gradio's ``special_args()`` scan stops at
    the first non-positional parameter, so a ``*, progress=...`` form is
    invisible to it and the live tracker is never injected. The default
    keeps direct/unit calls working without passing it; gradio injects the
    tracker itself, so ``progress`` must not appear in event ``inputs``
    lists.
    """
    if signal_transform is None:
        gr.Warning("Load a signal first (or click Load Example Signal)")
        return model, loaded_name
    if model is None or loaded_name != model_file:
        progress(0.1, desc="Loading model…")
        model = wrapper_model_load(model_file)
        loaded_name = model_file
    return model, loaded_name


def wrapper_run_inference(signal_transform, model, progress=gr.Progress()):
    """Run inference for the Analyze chain, with a progress step.

    Thin wrapper around the core ``model_infer`` so the ``progress`` kwarg
    stays out of that function's own (positional) API. Like ``ensure_model``,
    ``progress`` must stay a trailing plain default parameter (not
    keyword-only) so gradio's ``special_args()`` detects and injects it.
    """
    progress(0.6, desc="Running inference…")
    return model_infer(signal_transform, model)


def wrapper_load_single(
    signal_directory, signal_file, n_fft, hop_length, clip_dc, clip_low, clip_high
):
    """Wrapper to construct filepath from signal directory + signal file.

    Builds the transform args inline from the current slider values, so a
    Load always reflects them — there's no separate "Apply Transform
    Settings" step to forget to click.
    """
    if not signal_directory or not signal_file:
        return None
    transform_args = setup_stft_transform(
        n_fft, hop_length, clip_dc, clip_low, clip_high
    )
    return load_single(Path(signal_directory) / signal_file, transform_args)


def wrapper_load_multi(
    signal_directory,
    signal_1,
    signal_2,
    n_fft,
    hop_length,
    clip_dc,
    clip_low,
    clip_high,
):
    """Wrapper to construct list of filepaths from signal directory + signal files."""
    if not signal_directory or not signal_1 or not signal_2:
        return None
    transform_args = setup_stft_transform(
        n_fft, hop_length, clip_dc, clip_low, clip_high
    )
    return load_multi(
        [Path(signal_directory) / signal_1, Path(signal_directory) / signal_2],
        transform_args,
    )


def wrapper_load_example(n_fft, hop_length, clip_dc, clip_low, clip_high):
    """Wrapper building transform args inline for the example-signal loader."""
    transform_args = setup_stft_transform(
        n_fft, hop_length, clip_dc, clip_low, clip_high
    )
    return load_example_signal(transform_args)


def export_analysis(
    signal_transform,
    inference_output,
    model_file,
    n_fft,
    hop_length,
    clip_dc,
    clip_low,
    clip_high,
    threshold,
    view_mode,
):
    """Save the loaded spectrogram (+ mask, if inferred) as a ``.npz`` bundle.

    No ``stft_meta`` is passed to :func:`tokeye.export.analysis_bundle`: this
    tab loads arbitrary signal files with unknown sample rate, so pixel-centre
    time/frequency axes can't be derived here.

    The no-data path returns ``None`` (not ``gr.update()``, gradio's skip
    sentinel) so the ``gr.File`` download slot CLEARS instead of leaving a
    previous successful export visible as a stale link — same convention as
    ``handle_save_mask`` in ``tokeye.app.tabs.annotate``.
    """
    if signal_transform is None:
        gr.Warning("Load a signal first")
        return None

    params = {
        "model": model_file,
        "n_fft": n_fft,
        "hop": hop_length,
        "clip_dc": clip_dc,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "threshold": threshold,
        "view_mode": view_mode,
    }
    bundle = export.analysis_bundle(
        spectrogram=signal_transform,
        mask=inference_output,
        params=params,
        source="analyze",
    )
    out_dir = Path(tempfile.mkdtemp(prefix="tokeye-export-"))
    path = export.save_npz(out_dir / f"{export.default_stem('analysis')}.npz", bundle)
    return str(path)


def analyze_tab():
    # User Interface
    with gr.Column():
        ## Getting Started
        with gr.Accordion("Getting started — where to put your data", open=False):
            gr.Markdown(GETTING_STARTED_MD)

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
            visualize_out = gr.Image(label="Visualization", type="pil")

            save_export_btn = gr.Button("Save results (.npz)")
            export_out = gr.File(
                label="Download analysis (.npz)", interactive=False
            )

    # State variables
    model = gr.State()
    model_name = gr.State(None)
    signal_transform = gr.State()
    inference_output = gr.State()

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
        fn=wrapper_model_load_pair,
        inputs=[model_file],
        outputs=[model, model_name],
    )

    ## Signal — transform args are built inline from the live slider values,
    ## so a Load always reflects them (no separate "Apply Transform Settings"
    ## step to forget to click).
    load_single_btn.click(
        fn=wrapper_load_single,
        inputs=[
            signal_directory,
            signal_single,
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
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
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
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
        inputs=[n_fft, hop_length, clip_dc, clip_low_sld, clip_high_sld],
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

    ## Visualization — live re-render from cached state: every view control
    ## re-runs show_image against the already-computed inference_output, no
    ## re-inference needed.
    visualize_rerender_inputs = [
        view_mode,
        signal_transform,
        inference_output,
        out_1_chk,
        out_2_chk,
        vmin_sld,
        vmax_sld,
        threshold_sld,
    ]
    view_mode.change(
        fn=toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp],
    ).then(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )
    out_1_chk.change(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )
    out_2_chk.change(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )
    vmin_sld.release(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )
    vmax_sld.release(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )
    threshold_sld.release(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )

    ## One-click Analyze: ensure a model is loaded, run inference, visualize
    analyze_btn.click(
        fn=ensure_model,
        inputs=[model, model_name, model_file, signal_transform],
        outputs=[model, model_name],
    ).then(
        fn=wrapper_run_inference,
        inputs=[signal_transform, model],
        outputs=[inference_output],
    ).then(
        fn=show_image,
        inputs=visualize_rerender_inputs,
        outputs=[visualize_out],
    )

    ## Export
    save_export_btn.click(
        fn=export_analysis,
        inputs=[
            signal_transform,
            inference_output,
            model_file,
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
            threshold_sld,
            view_mode,
        ],
        outputs=[export_out],
    )
