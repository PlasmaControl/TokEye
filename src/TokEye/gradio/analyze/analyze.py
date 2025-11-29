import gradio as gr

from TokEye.gradio.utils.analyze import (
    compute_stft_pipeline,
    get_available_models,
    get_available_signals,
    handle_load,
    run_inference,
    swap_signal,
    update_visualization,
)


def channel_checkboxes():
    """Create ch0/ch1 checkbox pair."""
    with gr.Row():
        ch0 = gr.Checkbox(value=True, label="Channel 0")
        ch1 = gr.Checkbox(value=False, label="Channel 1")
    return ch0, ch1


def slider_pair(min_val, max_val, default_min, default_max, step, label_min, label_max):
    """Create paired min/max sliders."""
    with gr.Row():
        slider_min = gr.Slider(
            min_val, max_val, value=default_min, step=step, label=label_min
        )
        slider_max = gr.Slider(
            min_val, max_val, value=default_max, step=step, label=label_max
        )
    return slider_min, slider_max


def on_change_multi(components, fn, inputs, outputs):
    """Register same change handler for multiple components."""
    for comp in components:
        comp.change(fn=fn, inputs=inputs, outputs=outputs)


def analyze_tab():
    with gr.Column():
        with gr.Accordion("1. Load Model & Signal", open=True):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Model",
                    )
                with gr.Column():
                    signal_dropdown = gr.Dropdown(
                        choices=get_available_signals(),
                        label="Select Signal",
                    )
            load_btn = gr.Button("Load & Warmup", variant="primary")
            load_status = gr.Textbox(label="Status", lines=2, interactive=False)

        # Block 2: Transform
        with gr.Accordion("2. Transform Signal", open=True):
            with gr.Row():
                n_fft = gr.Slider(256, 2048, value=1024, step=256, label="N_FFT")
                hop_length = gr.Slider(64, 512, value=128, step=64, label="Hop Length")
                clip_dc = gr.Checkbox(value=True, label="Remove DC Bin")
            with gr.Row():
                percentile_low = gr.Slider(
                    0, 10, value=1.0, step=0.1, label="Clip Low %"
                )
                percentile_high = gr.Slider(
                    90, 100, value=99.0, step=0.1, label="Clip High %"
                )
            transform_btn = gr.Button("Compute STFT", variant="primary")
            transform_preview = gr.Image(label="Spectrogram Preview", type="pil")

        # Block 3: Inference
        with gr.Accordion("3. Model Inference", open=True):
            inference_btn = gr.Button("Run Inference", variant="primary")
            inference_status = gr.Textbox(label="Status", lines=2, interactive=False)

        # Block 4: Visualization
        with gr.Accordion("4. Visualization", open=True):
            view_mode = gr.Radio(
                choices=["Original", "Enhanced", "Mask", "Labels"],
                value="Enhanced",
                label="View Mode",
            )

            # Enhanced controls
            with gr.Group(visible=True) as enhanced_group:
                gr.Markdown("### Enhanced View")
                ch0_enh, ch1_enh = channel_checkboxes()
                clip_min, clip_max = slider_pair(
                    0, 1, 0, 1, 0.01, "Clip Min", "Clip Max"
                )

            # Mask controls
            with gr.Group(visible=False) as mask_group:
                gr.Markdown("### Mask View")
                ch0_mask, ch1_mask = channel_checkboxes()
                threshold_mask = gr.Slider(
                    0, 1, value=0.5, step=0.01, label="Threshold"
                )

            # Labels controls
            with gr.Group(visible=False) as labels_group:
                gr.Markdown("### Labels View")
                ch0_labels, ch1_labels = channel_checkboxes()
                threshold_labels = gr.Slider(
                    0, 1, value=0.5, step=0.01, label="Threshold"
                )
                min_size = gr.Slider(1, 500, value=50, step=1, label="Min Object Size")

            viz_output = gr.Image(label="Visualization", type="pil")

    # Event handlers
    load_btn.click(
        fn=handle_load,
        inputs=[model_dropdown, signal_dropdown],
        outputs=[model_state, signal_state, load_status],
    )

    signal_dropdown.change(
        fn=swap_signal,
        inputs=[signal_dropdown],
        outputs=[signal_state, load_status],
    )

    transform_btn.click(
        fn=compute_stft_pipeline,
        inputs=[
            signal_state,
            n_fft,
            hop_length,
            clip_dc,
            percentile_low,
            percentile_high,
        ],
        outputs=[spectrogram_state, transform_preview],
    )

    inference_btn.click(
        fn=run_inference,
        inputs=[spectrogram_state, model_state],
        outputs=[inference_state, inference_status],
    )

    # Toggle control visibility
    def toggle_controls(mode):
        return {
            enhanced_group: gr.update(visible=(mode == "Enhanced")),
            mask_group: gr.update(visible=(mode == "Mask")),
            labels_group: gr.update(visible=(mode == "Labels")),
        }

    view_mode.change(
        fn=toggle_controls,
        inputs=[view_mode],
        outputs=[enhanced_group, mask_group, labels_group],
    )

    # Visualization updates - all controls trigger this
    all_viz_inputs = [
        view_mode,
        spectrogram_state,
        inference_state,
        ch0_enh,
        ch1_enh,
        clip_min,
        clip_max,
        ch0_mask,
        ch1_mask,
        threshold_mask,
        ch0_labels,
        ch1_labels,
        threshold_labels,
        min_size,
    ]

    on_change_multi(
        [
            view_mode,
            ch0_enh,
            ch1_enh,
            clip_min,
            clip_max,
            ch0_mask,
            ch1_mask,
            threshold_mask,
            ch0_labels,
            ch1_labels,
            threshold_labels,
            min_size,
        ],
        fn=update_visualization,
        inputs=all_viz_inputs,
        outputs=[viz_output],
    )
