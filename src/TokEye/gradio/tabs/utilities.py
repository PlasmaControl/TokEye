"""
Utilities Tab for TokEye

This module provides audio conversion tools and .npy file inspection utilities.
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import io
import tempfile


# ============================================================================
# Audio Conversion Functions
# ============================================================================

def load_audio_file(audio_file) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """
    Load audio file and extract waveform.

    Returns:
        (waveform_array, sample_rate, info_text)
    """
    if audio_file is None:
        return None, None, "No file uploaded"

    try:
        import librosa
        import soundfile as sf

        # Try loading with librosa first (handles more formats)
        try:
            waveform, sample_rate = librosa.load(audio_file, sr=None, mono=False)

            # Ensure mono
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)

        except Exception:
            # Fallback to soundfile
            waveform, sample_rate = sf.read(audio_file)

            # Ensure mono
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

        info = f"""
**Audio File Information:**
- Sample rate: {sample_rate:,} Hz
- Duration: {len(waveform) / sample_rate:.2f} seconds
- Samples: {len(waveform):,}
- Data type: {waveform.dtype}
- Min: {waveform.min():.4f}
- Max: {waveform.max():.4f}
- Mean: {waveform.mean():.4f}
"""

        return waveform, sample_rate, info

    except ImportError:
        return None, None, "Error: librosa or soundfile not installed. Install with: pip install librosa soundfile"
    except Exception as e:
        return None, None, f"Error loading audio: {str(e)}"


def convert_audio_to_npy(
    waveform: Optional[np.ndarray],
    sample_rate: Optional[int],
    normalize: bool = True
) -> Tuple[Optional[str], str]:
    """
    Convert audio waveform to .npy file.

    Args:
        waveform: Audio waveform array
        sample_rate: Sample rate
        normalize: Whether to normalize to [-1, 1]

    Returns:
        (filepath, status_text)
    """
    if waveform is None:
        return None, "No audio loaded"

    try:
        # Normalize if requested
        if normalize:
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val

        # Save to temporary directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"audio_converted_{timestamp}.npy"

        np.save(filepath, waveform)

        status = f"""
**Conversion Successful:**
- Output file: {filepath}
- Shape: {waveform.shape}
- Sample rate: {sample_rate} Hz (saved separately in metadata)
- Normalized: {normalize}

Note: Sample rate information should be stored separately.
Consider saving as: {filepath.stem}_sr{sample_rate}.npy
"""

        gr.Info(f"Converted to {filepath}")
        return str(filepath), status

    except Exception as e:
        return None, f"Error during conversion: {str(e)}"


# ============================================================================
# Audio Recording Functions
# ============================================================================

def process_recorded_audio(audio_data) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """
    Process audio from gr.Audio component.

    Args:
        audio_data: Tuple of (sample_rate, waveform) from gr.Audio

    Returns:
        (waveform_array, sample_rate, info_text)
    """
    if audio_data is None:
        return None, None, "No recording available"

    try:
        # gr.Audio returns (sample_rate, waveform)
        sample_rate, waveform = audio_data

        # Convert to float and normalize
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768.0
        elif waveform.dtype == np.int32:
            waveform = waveform.astype(np.float32) / 2147483648.0
        else:
            waveform = waveform.astype(np.float32)

        # Ensure mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        info = f"""
**Recording Information:**
- Sample rate: {sample_rate:,} Hz
- Duration: {len(waveform) / sample_rate:.2f} seconds
- Samples: {len(waveform):,}
- Data type: {waveform.dtype}
"""

        return waveform, sample_rate, info

    except Exception as e:
        return None, None, f"Error processing recording: {str(e)}"


# ============================================================================
# .npy File Inspector
# ============================================================================

def inspect_npy_file(file) -> str:
    """
    Inspect .npy file and display detailed statistics.

    Returns:
        info_text (formatted string)
    """
    if file is None:
        return "No file uploaded"

    try:
        # Load array
        arr = np.load(file.name)

        # Basic info
        info = f"""
# NumPy Array Information

## Basic Properties
- **Shape:** {arr.shape}
- **Dimensions:** {arr.ndim}D
- **Data type:** {arr.dtype}
- **Total elements:** {arr.size:,}
- **Memory size:** {arr.nbytes / 1024:.2f} KB ({arr.nbytes / (1024*1024):.4f} MB)

## Statistical Summary
- **Min value:** {arr.min():.6f}
- **Max value:** {arr.max():.6f}
- **Mean:** {arr.mean():.6f}
- **Std deviation:** {arr.std():.6f}
- **Median:** {np.median(arr):.6f}

## Data Distribution
- **25th percentile:** {np.percentile(arr, 25):.6f}
- **50th percentile (median):** {np.percentile(arr, 50):.6f}
- **75th percentile:** {np.percentile(arr, 75):.6f}

## Special Values
- **Number of zeros:** {np.count_nonzero(arr == 0):,}
- **Number of NaNs:** {np.count_nonzero(np.isnan(arr)):,}
- **Number of Infs:** {np.count_nonzero(np.isinf(arr)):,}
"""

        # Add dimension-specific info
        if arr.ndim == 1:
            info += f"\n## 1D Array Properties\n- Length: {len(arr):,} samples\n"
        elif arr.ndim == 2:
            info += f"\n## 2D Array Properties\n- Height: {arr.shape[0]}\n- Width: {arr.shape[1]}\n"
        elif arr.ndim == 3:
            info += f"\n## 3D Array Properties\n- Channels: {arr.shape[0]}\n- Height: {arr.shape[1]}\n- Width: {arr.shape[2]}\n"

        return info

    except Exception as e:
        return f"Error inspecting file: {str(e)}"


# ============================================================================
# Batch Conversion
# ============================================================================

def batch_convert_audio_files(files: List) -> Tuple[str, List[str]]:
    """
    Convert multiple audio files to .npy format.

    Args:
        files: List of audio files

    Returns:
        (status_text, list_of_output_files)
    """
    if not files:
        return "No files provided", []

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return "Error: librosa and soundfile required. Install with: pip install librosa soundfile", []

    output_dir = Path("outputs/batch_conversion")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    success_count = 0
    failed_files = []

    for i, audio_file in enumerate(files):
        try:
            # Load audio
            try:
                waveform, sample_rate = librosa.load(audio_file.name, sr=None, mono=True)
            except Exception:
                waveform, sample_rate = sf.read(audio_file.name)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)

            # Normalize
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val

            # Generate output filename
            input_name = Path(audio_file.name).stem
            output_file = output_dir / f"{input_name}_sr{sample_rate}.npy"

            # Save
            np.save(output_file, waveform)

            output_files.append(str(output_file))
            success_count += 1

        except Exception as e:
            failed_files.append(f"{audio_file.name}: {str(e)}")

    # Generate status report
    status = f"""
**Batch Conversion Complete**

- **Total files:** {len(files)}
- **Successful:** {success_count}
- **Failed:** {len(failed_files)}
- **Output directory:** {output_dir}

"""

    if failed_files:
        status += "\n**Failed files:**\n"
        for failure in failed_files:
            status += f"- {failure}\n"

    if success_count > 0:
        gr.Info(f"Successfully converted {success_count}/{len(files)} files")

    return status, output_files


# ============================================================================
# Gradio Interface
# ============================================================================

def utilities_tab():
    """Create the utilities tab interface."""

    with gr.Column() as tab:
        gr.Markdown("# TokEye Utilities")
        gr.Markdown("Audio conversion tools and file inspection utilities.")

        # ====================================================================
        # Section 1: Audio File Conversion
        # ====================================================================
        with gr.Accordion("Audio File Conversion", open=True):
            gr.Markdown("### Convert Audio Files to NumPy Arrays")
            gr.Markdown("Upload audio files (.wav, .mp3, .ogg, etc.) and convert to .npy format for analysis.")

            # State variables
            audio_waveform_state = gr.State(None)
            audio_sr_state = gr.State(None)

            with gr.Row():
                with gr.Column():
                    audio_file_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        sources=["upload"]
                    )

                    load_audio_btn = gr.Button("Load Audio", variant="primary")

                    audio_info = gr.Textbox(
                        label="Audio Information",
                        lines=10,
                        interactive=False
                    )

                with gr.Column():
                    normalize_checkbox = gr.Checkbox(
                        label="Normalize to [-1, 1]",
                        value=True
                    )

                    convert_audio_btn = gr.Button("Convert to .npy", variant="primary")

                    conversion_status = gr.Textbox(
                        label="Conversion Status",
                        lines=10,
                        interactive=False
                    )

                    download_npy_file = gr.File(
                        label="Download Converted File",
                        interactive=False
                    )

        # ====================================================================
        # Section 2: Audio Recording
        # ====================================================================
        with gr.Accordion("Audio Recording", open=False):
            gr.Markdown("### Record Audio Directly")
            gr.Markdown("Record audio using your microphone and convert to .npy format.")

            # State variables
            recorded_waveform_state = gr.State(None)
            recorded_sr_state = gr.State(None)

            with gr.Row():
                with gr.Column():
                    audio_recorder = gr.Audio(
                        label="Record Audio",
                        sources=["microphone"],
                        type="numpy"
                    )

                    process_recording_btn = gr.Button("Process Recording", variant="primary")

                    recording_info = gr.Textbox(
                        label="Recording Information",
                        lines=6,
                        interactive=False
                    )

                with gr.Column():
                    recording_playback = gr.Audio(
                        label="Playback Preview",
                        type="numpy",
                        interactive=False
                    )

                    convert_recording_btn = gr.Button("Convert to .npy", variant="primary")

                    recording_conversion_status = gr.Textbox(
                        label="Conversion Status",
                        lines=6,
                        interactive=False
                    )

                    download_recording_file = gr.File(
                        label="Download Converted Recording",
                        interactive=False
                    )

        # ====================================================================
        # Section 3: .npy File Inspector
        # ====================================================================
        with gr.Accordion(".npy File Inspector", open=False):
            gr.Markdown("### Inspect NumPy Array Files")
            gr.Markdown("Upload a .npy file to view detailed statistics and properties.")

            with gr.Row():
                with gr.Column(scale=1):
                    npy_file_input = gr.File(
                        label="Upload .npy File",
                        file_types=[".npy"]
                    )

                    inspect_btn = gr.Button("Inspect File", variant="primary")

                with gr.Column(scale=2):
                    inspection_output = gr.Markdown("*No file inspected*")

        # ====================================================================
        # Section 4: Batch Conversion
        # ====================================================================
        with gr.Accordion("Batch Audio Conversion", open=False):
            gr.Markdown("### Convert Multiple Audio Files")
            gr.Markdown("Upload multiple audio files and convert them all at once.")

            batch_files_input = gr.File(
                label="Upload Audio Files",
                file_count="multiple",
                file_types=[".wav", ".mp3", ".ogg", ".flac", ".m4a"]
            )

            batch_convert_btn = gr.Button("Convert All Files", variant="primary")

            batch_status = gr.Textbox(
                label="Batch Conversion Status",
                lines=10,
                interactive=False
            )

            batch_output_list = gr.File(
                label="Converted Files",
                file_count="multiple",
                interactive=False
            )

        # ====================================================================
        # Event Handlers
        # ====================================================================

        # Load audio file
        load_audio_btn.click(
            fn=load_audio_file,
            inputs=[audio_file_input],
            outputs=[audio_waveform_state, audio_sr_state, audio_info]
        )

        # Convert audio to .npy
        def handle_audio_conversion(waveform, sample_rate, normalize):
            filepath, status = convert_audio_to_npy(waveform, sample_rate, normalize)
            if filepath:
                return status, filepath
            else:
                return status, None

        convert_audio_btn.click(
            fn=handle_audio_conversion,
            inputs=[audio_waveform_state, audio_sr_state, normalize_checkbox],
            outputs=[conversion_status, download_npy_file]
        )

        # Process recording
        process_recording_btn.click(
            fn=process_recorded_audio,
            inputs=[audio_recorder],
            outputs=[recorded_waveform_state, recorded_sr_state, recording_info]
        )

        # Update playback preview when recording is processed
        def update_playback(waveform, sample_rate):
            if waveform is not None and sample_rate is not None:
                return (sample_rate, waveform)
            return None

        process_recording_btn.click(
            fn=update_playback,
            inputs=[recorded_waveform_state, recorded_sr_state],
            outputs=[recording_playback]
        )

        # Convert recording to .npy
        def handle_recording_conversion(waveform, sample_rate):
            filepath, status = convert_audio_to_npy(waveform, sample_rate, normalize=True)
            if filepath:
                return status, filepath
            else:
                return status, None

        convert_recording_btn.click(
            fn=handle_recording_conversion,
            inputs=[recorded_waveform_state, recorded_sr_state],
            outputs=[recording_conversion_status, download_recording_file]
        )

        # Inspect .npy file
        inspect_btn.click(
            fn=inspect_npy_file,
            inputs=[npy_file_input],
            outputs=[inspection_output]
        )

        # Batch conversion
        def handle_batch_conversion(files):
            status, output_files = batch_convert_audio_files(files)
            if output_files:
                return status, output_files
            else:
                return status, None

        batch_convert_btn.click(
            fn=handle_batch_conversion,
            inputs=[batch_files_input],
            outputs=[batch_status, batch_output_list]
        )

    return tab
