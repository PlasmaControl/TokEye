"""Tests for the Annotate tab (src/tokeye/app/tabs/annotate.py).

Covers four defects fixed together:
1. Float backdrops rendering near-black (bad normalization).
2. O(H*W) pure-Python compositing hanging on realistic image sizes.
3. Non-red brush strokes being silently dropped during mask extraction.
4. No browser-reachable download for the saved mask.

These tests run offline (no browser, no Gradio server) by calling the
module-level helper functions directly, mirroring the value shapes that
`gr.ImageEditor(type="pil", image_mode="RGBA")` actually produces:
a dict with "background", "layers" (list of RGBA PIL Images), and
"composite" (a PIL Image), as confirmed by the installed gradio 5.49.1
`ImageEditor.preprocess`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tokeye.app.tabs import annotate

# ============================================================================
# to_display_uint8
# ============================================================================


def test_to_display_uint8_stretches_float_log_spectrogram():
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 9, size=(32, 32)).astype(np.float32)

    out = annotate.to_display_uint8(arr)

    assert out.dtype == np.uint8
    assert int(out.max()) - int(out.min()) > 200


def test_to_display_uint8_uint8_passthrough():
    arr = (np.arange(64, dtype=np.uint8)).reshape(8, 8)

    out = annotate.to_display_uint8(arr)

    assert out.dtype == np.uint8
    assert np.array_equal(out, arr)


def test_to_display_uint8_flat_image_does_not_crash():
    arr = np.full((10, 10), 3.5, dtype=np.float64)

    out = annotate.to_display_uint8(arr)

    assert out.dtype == np.uint8
    assert np.array_equal(out, np.zeros((10, 10), dtype=np.uint8))


def test_to_display_uint8_all_nan_is_zeros_without_warnings():
    arr = np.full((6, 6), np.nan, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = annotate.to_display_uint8(arr)

    assert out.dtype == np.uint8
    assert np.array_equal(out, np.zeros((6, 6), dtype=np.uint8))


# ============================================================================
# create_composite_image (vectorized overlay equivalence)
# ============================================================================


def _reference_composite(
    backdrop_img: Image.Image, mask_arr: np.ndarray
) -> Image.Image:
    """Tiny reference loop mirroring the pre-fix putpixel semantics."""
    composite = backdrop_img.copy().convert("RGBA")
    if mask_arr.max() > 0:
        red_overlay = Image.new("RGBA", composite.size)
        for x in range(composite.size[0]):
            for y in range(composite.size[1]):
                if x < mask_arr.shape[1] and y < mask_arr.shape[0]:
                    if mask_arr[y, x] > 0:
                        red_overlay.putpixel((x, y), (255, 0, 0, 128))
                    else:
                        red_overlay.putpixel((x, y), (0, 0, 0, 0))
        composite = Image.alpha_composite(composite, red_overlay)
    return composite


def test_create_composite_image_matches_reference_loop():
    backdrop_rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    backdrop_rgba[..., 0] = 50
    backdrop_rgba[..., 1] = 60
    backdrop_rgba[..., 2] = 70
    backdrop_rgba[..., 3] = 255
    backdrop_img = Image.fromarray(backdrop_rgba, mode="RGBA")

    mask_arr = np.zeros((8, 8), dtype=np.uint8)
    mask_arr[2:5, 3:6] = 255

    expected = _reference_composite(backdrop_img, mask_arr)
    actual = annotate.create_composite_image(backdrop_img, mask_arr)

    assert np.array_equal(np.array(actual), np.array(expected))

    actual_arr = np.array(actual)
    # Masked pixels are red-tinted (differ from the plain backdrop).
    assert not np.array_equal(actual_arr[3, 4], backdrop_rgba[3, 4])
    assert actual_arr[3, 4, 0] > actual_arr[3, 4, 2]  # red channel boosted
    # Unmasked pixels are untouched.
    assert np.array_equal(actual_arr[0, 0], backdrop_rgba[0, 0])


def test_create_composite_image_no_mask_is_unchanged():
    backdrop_rgba = np.full((6, 6, 4), 30, dtype=np.uint8)
    backdrop_rgba[..., 3] = 255
    backdrop_img = Image.fromarray(backdrop_rgba, mode="RGBA")
    mask_arr = np.zeros((6, 6), dtype=np.uint8)

    actual = annotate.create_composite_image(backdrop_img, mask_arr)

    assert np.array_equal(np.array(actual), backdrop_rgba)


def test_create_composite_image_mask_smaller_than_backdrop():
    """A mask smaller than the backdrop only tints its own (top-left) region."""
    backdrop_rgba = np.full((8, 8, 4), 30, dtype=np.uint8)
    backdrop_rgba[..., 3] = 255
    backdrop_img = Image.fromarray(backdrop_rgba, mode="RGBA")

    mask_arr = np.full((4, 4), 255, dtype=np.uint8)  # smaller than 8x8

    actual = annotate.create_composite_image(backdrop_img, mask_arr)
    actual_arr = np.array(actual)

    assert actual.size == backdrop_img.size
    # In-mask region is red-tinted.
    assert (actual_arr[:4, :4, 0] > actual_arr[:4, :4, 2]).all()
    # Everything beyond the mask extent is untouched.
    assert np.array_equal(actual_arr[4:, :], backdrop_rgba[4:, :])
    assert np.array_equal(actual_arr[:4, 4:], backdrop_rgba[:4, 4:])


def test_create_composite_image_mask_larger_than_backdrop():
    """A mask larger than the backdrop is cropped, not resized or crashed on."""
    backdrop_rgba = np.full((6, 6, 4), 30, dtype=np.uint8)
    backdrop_rgba[..., 3] = 255
    backdrop_img = Image.fromarray(backdrop_rgba, mode="RGBA")

    mask_arr = np.zeros((10, 10), dtype=np.uint8)  # larger than 6x6
    mask_arr[1:3, 1:3] = 255  # in-bounds stroke
    mask_arr[8:, 8:] = 255  # out-of-bounds stroke - must be ignored

    actual = annotate.create_composite_image(backdrop_img, mask_arr)
    actual_arr = np.array(actual)

    assert actual.size == backdrop_img.size
    # In-bounds masked pixels are red-tinted.
    assert (actual_arr[1:3, 1:3, 0] > actual_arr[1:3, 1:3, 2]).all()
    # Pixels not covered by an in-bounds stroke are untouched.
    assert np.array_equal(actual_arr[4:, 4:], backdrop_rgba[4:, 4:])


# ============================================================================
# extract_mask_from_canvas (color-agnostic, layers-first)
# ============================================================================


def test_extract_mask_from_canvas_layers_green_stroke():
    backdrop_arr = np.full((8, 8), 40, dtype=np.uint8)
    background_img = annotate.numpy_to_pil(backdrop_arr)

    layer = np.zeros((8, 8, 4), dtype=np.uint8)
    layer[2:4, 2:4] = [0, 255, 0, 200]  # green stroke
    layer_img = Image.fromarray(layer, mode="RGBA")

    canvas_output = {
        "background": background_img,
        "layers": [layer_img],
        "composite": background_img,
    }

    mask = annotate.extract_mask_from_canvas(canvas_output, backdrop_arr)

    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:4, 2:4] = 255
    assert mask.dtype == np.uint8
    assert np.array_equal(mask, expected)


def test_extract_mask_from_canvas_composite_fallback_red_stroke():
    """Dict without layers falls back to the RGB-diff-vs-backdrop path."""
    backdrop_arr = np.full((8, 8), 40, dtype=np.uint8)
    background_img = annotate.numpy_to_pil(backdrop_arr)

    composite_arr = annotate.pil_to_numpy(background_img).copy()
    composite_arr[3:5, 3:5] = [255, 0, 0, 255]  # red stroke
    composite_img = Image.fromarray(composite_arr, mode="RGBA")

    canvas_output = {
        "background": background_img,
        "layers": [],
        "composite": composite_img,
    }

    mask = annotate.extract_mask_from_canvas(canvas_output, backdrop_arr)

    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[3:5, 3:5] = 255
    assert np.array_equal(mask, expected)


def test_extract_mask_from_canvas_composite_fallback_float_backdrop():
    """Fallback diff must normalize BOTH sides for float (log-scale) backdrops.

    Pre-fix, the canvas (normalized for display) was diffed against the RAW
    float backdrop, so extraction was broken whenever the input .npy was a
    log-scaled spectrogram. Draw a stroke on the normalized composite and
    assert the diff isolates exactly the stroke.
    """
    # Deterministic gradient spanning typical log-spectrogram values 0-9.
    backdrop_arr = np.linspace(0.0, 9.0, 64, dtype=np.float32).reshape(8, 8)
    background_img = annotate.numpy_to_pil(backdrop_arr)  # normalized display

    composite_arr = annotate.pil_to_numpy(background_img).copy()
    composite_arr[2:4, 5:7] = [255, 0, 0, 255]  # red stroke on the display
    composite_img = Image.fromarray(composite_arr, mode="RGBA")

    canvas_output = {
        "background": background_img,
        "layers": [],
        "composite": composite_img,
    }

    mask = annotate.extract_mask_from_canvas(canvas_output, backdrop_arr)

    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:4, 5:7] = 255
    assert np.array_equal(mask, expected)


@pytest.mark.parametrize(
    "color",
    [
        pytest.param([0, 0, 255, 180], id="blue"),
        pytest.param([255, 255, 255, 220], id="white"),
    ],
)
def test_extract_mask_from_canvas_layers_non_red_colors(color):
    backdrop_arr = np.full((8, 8), 40, dtype=np.uint8)
    background_img = annotate.numpy_to_pil(backdrop_arr)

    layer = np.zeros((8, 8, 4), dtype=np.uint8)
    layer[5:7, 5:7] = color
    layer_img = Image.fromarray(layer, mode="RGBA")

    canvas_output = {
        "background": background_img,
        "layers": [layer_img],
        "composite": background_img,
    }

    mask = annotate.extract_mask_from_canvas(canvas_output, backdrop_arr)

    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[5:7, 5:7] = 255
    assert np.array_equal(mask, expected)


# ============================================================================
# save handler + browser download
# ============================================================================


def test_handle_save_mask_writes_file_and_returns_downloadable_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    backdrop_arr = np.full((6, 6), 20, dtype=np.uint8)
    background_img = annotate.numpy_to_pil(backdrop_arr)
    layer = np.zeros((6, 6, 4), dtype=np.uint8)
    layer[1:3, 1:3] = [255, 0, 0, 255]
    canvas_output = {
        "background": background_img,
        "layers": [Image.fromarray(layer, mode="RGBA")],
        "composite": background_img,
    }

    status, download = annotate.handle_save_mask(
        canvas_output, backdrop_arr, "shot123.npy", "npy"
    )

    saved_path = Path("annotations") / "shot123_mask.npy"
    assert saved_path.exists()
    assert "saved" in status.lower()
    assert Path(download).resolve() == saved_path.resolve()


def test_handle_save_mask_missing_canvas_clears_download(tmp_path, monkeypatch):
    """Failure must return None for the gr.File slot (clears the component).

    A bare gr.update() is gradio's skip sentinel - the wire payload carries
    no value key, so a previous successful save's file would stay visible
    after a later failed save (stale download link).
    """
    monkeypatch.chdir(tmp_path)

    status, download = annotate.handle_save_mask(None, None, None, "npy")

    assert "error" in status.lower() or "no annotation" in status.lower()
    assert download is None


def test_failed_save_after_successful_save_clears_stale_download(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    backdrop_arr = np.full((6, 6), 20, dtype=np.uint8)
    background_img = annotate.numpy_to_pil(backdrop_arr)
    layer = np.zeros((6, 6, 4), dtype=np.uint8)
    layer[1:3, 1:3] = [255, 0, 0, 255]
    canvas_output = {
        "background": background_img,
        "layers": [Image.fromarray(layer, mode="RGBA")],
        "composite": background_img,
    }

    # Successful save populates the download slot with the file path.
    _, download_ok = annotate.handle_save_mask(
        canvas_output, backdrop_arr, "shot123.npy", "npy"
    )
    assert Path(download_ok).exists()

    # A subsequent failed save must CLEAR the slot, not skip the update.
    status, download_fail = annotate.handle_save_mask(
        None, backdrop_arr, "shot123.npy", "npy"
    )
    assert "error" in status.lower()
    assert download_fail is None
