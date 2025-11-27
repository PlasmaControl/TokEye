import sys
import io
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import joblib

import logging

logger = logging.getLogger(__name__)

from .utils.configuration import (
    load_settings,
    setup_directory,
)

default_settings = {
    "input_dir": Path("data/cache/step_5a_combine_spectrogram"),
    "output_dir": Path("data/cache/step_5b_manual_fix_spectrogram"),
    "frame_info_csv": Path("data/frame_info.csv"),
    "percentile_low": 1,
    "percentile_high": 99,
    "mask_alpha": 0.5,
    "mask_color": (255, 0, 0),  # Red
    "default_brush_size": 10,
}


class SpectrogramEditor:
    """GUI application for manually editing spectrogram masks."""

    # UI Constants
    MINIMAP_WIDTH = 800  # Wider to span bottom
    MINIMAP_HEIGHT = 120  # Slightly shorter
    DEFAULT_CANVAS_HEIGHT = 600
    DEFAULT_VIEWPORT_WIDTH = 1200

    def __init__(self, settings: dict):
        self.settings = settings
        self.root = tk.Tk()
        self.root.title("Spectrogram Mask Editor")

        # Data
        self.frame_info = self.load_frame_info()
        self.shot_numbers = sorted(self.frame_info["shotn"].unique().tolist())
        self.current_shot = None
        self.current_channel = None

        # Image data
        self.spectrogram = None  # Original spectrogram data
        self.mask = None  # Editable mask data (H, W)
        self.base_img_rgb = None  # Cached base spectrogram image (no mask overlay)
        self.display_image = None  # PIL Image for display
        self.photo_image = None  # ImageTk.PhotoImage for canvas

        # Canvas parameters
        self.canvas_height = self.DEFAULT_CANVAS_HEIGHT
        self.viewport_width = self.DEFAULT_VIEWPORT_WIDTH
        self.zoom_factor = 1.0

        # Drawing state
        self.drawing = False
        self.draw_mode = tk.StringVar(value="draw")  # 'draw' or 'erase'
        self.brush_size = tk.IntVar(value=settings["default_brush_size"])
        self.last_x = None
        self.last_y = None

        # Track if mask has been modified
        self.mask_modified = False
        self.current_stroke_modified = False  # Track if current stroke changed anything

        # Undo stack
        self.undo_stack = []  # Stack of previous mask states
        self.max_undo_levels = 20  # Maximum number of undo levels

        # Stroke preview
        self.stroke_preview_mask = (
            None  # Temporary mask showing current stroke in progress
        )
        self.cursor_position = None  # Current mouse position for brush cursor
        self.cursor_circle_item = None  # Canvas item ID for cursor circle
        self._viewport_preview_photo = None  # Cached viewport preview photo

        # Frame skipping for performance
        self._frame_counter = 0
        self._frame_skip = 2  # Only update every N frames during drag

        # Setup output directory
        setup_directory(
            path=self.settings["output_dir"],
            overwrite=False,
        )

        # Build GUI
        self.build_gui()

    def load_frame_info(self) -> pd.DataFrame:
        """Load frame_info.csv."""
        csv_path = self.settings["frame_info_csv"]
        logger.info(f"Loading frame info from {csv_path}")
        return pd.read_csv(csv_path)

    def build_gui(self):
        """Build the complete GUI."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Navigation panel
        self.build_navigation_panel(main_frame)

        # Middle: Canvas display with scrollbar
        self.build_canvas_panel(main_frame)

        # Bottom left: Tool panel
        self.build_tool_panel(main_frame)

        # Bottom center: Minimap panel
        self.build_minimap_panel(main_frame)

    def build_navigation_panel(self, parent):
        """Build the navigation panel at the top."""
        nav_frame = ttk.LabelFrame(parent, text="Navigation", padding=10)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Shot selection
        ttk.Label(nav_frame, text="Shot:").pack(side=tk.LEFT, padx=5)
        self.shot_combo = ttk.Combobox(
            nav_frame,
            values=[str(s) for s in self.shot_numbers],
            width=15,
            state="readonly",
        )
        self.shot_combo.pack(side=tk.LEFT, padx=5)
        if self.shot_numbers:
            self.shot_combo.current(0)

        # Channel selection
        ttk.Label(nav_frame, text="Channel:").pack(side=tk.LEFT, padx=5)
        self.channel_combo = ttk.Combobox(nav_frame, width=10, state="readonly")
        self.channel_combo.pack(side=tk.LEFT, padx=5)

        # Load button
        load_btn = ttk.Button(nav_frame, text="Load", command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=5)

        # Previous/Next buttons
        prev_btn = ttk.Button(nav_frame, text="Previous", command=self.load_previous)
        prev_btn.pack(side=tk.LEFT, padx=5)

        next_btn = ttk.Button(nav_frame, text="Next", command=self.load_next)
        next_btn.pack(side=tk.LEFT, padx=5)

        # Update channel list when shot changes
        self.shot_combo.bind("<<ComboboxSelected>>", self.update_channel_list)

        # Initialize channel list
        self.update_channel_list()

    def build_canvas_panel(self, parent):
        """Build the main canvas panel."""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas with scrollbar
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="black",
            height=self.canvas_height,
            width=self.viewport_width,
        )

        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.h_scrollbar.pack(side=tk.TOP, fill=tk.X)

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind(
            "<Motion>", self.on_mouse_move
        )  # Track mouse for cursor circle
        self.canvas.bind(
            "<Leave>", self.on_mouse_leave
        )  # Clear cursor when leaving canvas

        # Bind keyboard shortcuts
        self.root.bind("<Control-s>", lambda e: self.save_mask())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<bracketleft>", lambda e: self._adjust_brush_size(-5))
        self.root.bind("<bracketright>", lambda e: self._adjust_brush_size(5))
        self.root.bind("<d>", lambda e: self.draw_mode.set("draw"))
        self.root.bind("<e>", lambda e: self.draw_mode.set("erase"))

    def build_tool_panel(self, parent):
        """Build the tool panel at the bottom."""
        tool_frame = ttk.LabelFrame(parent, text="Tools", padding=10)
        tool_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Draw/Erase mode
        mode_frame = ttk.Frame(tool_frame)
        mode_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Draw", variable=self.draw_mode, value="draw"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Erase", variable=self.draw_mode, value="erase"
        ).pack(side=tk.LEFT, padx=5)

        # Brush size slider
        brush_frame = ttk.Frame(tool_frame)
        brush_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        brush_slider = ttk.Scale(
            brush_frame,
            from_=1,
            to=50,
            variable=self.brush_size,
            orient=tk.HORIZONTAL,
            length=200,
        )
        brush_slider.pack(side=tk.LEFT, padx=5)

        brush_label = ttk.Label(brush_frame, textvariable=self.brush_size, width=3)
        brush_label.pack(side=tk.LEFT, padx=5)

        # Save button
        save_btn = ttk.Button(tool_frame, text="Save", command=self.save_mask)
        save_btn.pack(side=tk.RIGHT, padx=10)

        # Restore Original button
        restore_btn = ttk.Button(
            tool_frame, text="Restore Original", command=self.restore_original
        )
        restore_btn.pack(side=tk.RIGHT, padx=10)

    def build_minimap_panel(self, parent):
        """Build the minimap panel at the bottom center."""
        minimap_frame = ttk.LabelFrame(parent, text="Overview", padding=5)
        minimap_frame.pack(side=tk.BOTTOM, padx=5, pady=5)

        # Container to center the canvas
        canvas_container = ttk.Frame(minimap_frame)
        canvas_container.pack()

        self.minimap_canvas = tk.Canvas(
            canvas_container,
            width=self.MINIMAP_WIDTH,
            height=self.MINIMAP_HEIGHT,
            bg="gray20",
        )
        self.minimap_canvas.pack()

        # Edited status label
        self.edited_status_label = ttk.Label(
            minimap_frame, text="", foreground="orange"
        )
        self.edited_status_label.pack(side=tk.BOTTOM, pady=2)

        # Viewport indicator (will be drawn when data is loaded)
        self.viewport_rect = None

    def update_channel_list(self, event=None):
        """Update the channel dropdown based on selected shot."""
        shot_str = self.shot_combo.get()
        if not shot_str:
            return

        shotn = int(shot_str)

        # Find available channels for this shot
        channels = self.get_available_channels(shotn)

        self.channel_combo["values"] = [str(c) for c in channels]
        if channels:
            self.channel_combo.current(0)

    def _get_file_paths(self, shotn: int, channel_idx: int) -> tuple[Path, Path]:
        """
        Get file paths for spectrogram and mask.

        Args:
            shotn: Shot number
            channel_idx: Channel index

        Returns:
            Tuple of (spectrogram_path, mask_path)
        """
        img_path = self.settings["input_dir"] / f"{shotn}_{channel_idx}.joblib"
        mask_path = self.settings["input_dir"] / f"{shotn}_{channel_idx}_mask.joblib"
        return img_path, mask_path

    def get_available_channels(self, shotn: int) -> list[int]:
        """Get list of available channels for a shot."""
        channels = []
        channel_idx = 0

        # Check for available files
        while True:
            img_path, mask_path = self._get_file_paths(shotn, channel_idx)

            if img_path.exists() and mask_path.exists():
                channels.append(channel_idx)
                channel_idx += 1
            else:
                break

        return channels

    def load_data(self, source="auto"):
        """
        Load spectrogram and mask data for the selected shot and channel.

        Args:
            source: 'auto' (prefer edited), 'input' (original), or 'output' (edited only)
        """
        shot_str = self.shot_combo.get()
        channel_str = self.channel_combo.get()

        if not shot_str or not channel_str:
            messagebox.showwarning("Warning", "Please select both shot and channel")
            return

        # Check if current mask has been modified
        if self.mask_modified:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "Current mask has been modified. Save before loading new data?",
            )
            if response is True:  # Yes
                self.save_mask()
            elif response is None:  # Cancel
                return

        shotn = int(shot_str)
        channel_idx = int(channel_str)

        logger.info(f"Loading shot {shotn}, channel {channel_idx} from {source}")

        try:
            # Load spectrogram and mask
            self.current_shot = shotn
            self.current_channel = channel_idx

            img_path, mask_path = self._get_file_paths(shotn, channel_idx)

            # Determine which mask to load based on source parameter
            if source == "input":
                # Always load from input directory (original)
                pass  # mask_path is already input directory
            elif source == "output":
                # Load from output directory only
                mask_path = (
                    self.settings["output_dir"] / f"{shotn}_{channel_idx}_mask.joblib"
                )
            else:  # source == 'auto'
                # Prefer output directory if it exists
                edited_mask_path = (
                    self.settings["output_dir"] / f"{shotn}_{channel_idx}_mask.joblib"
                )
                if edited_mask_path.exists():
                    mask_path = edited_mask_path

            # Load data
            self.spectrogram = joblib.load(
                img_path
            )  # Shape: (F, W) - already magnitude
            mask_data = joblib.load(mask_path)  # Shape: (H, W) or (H, W, 1)

            logger.info(f"Loaded spectrogram shape: {self.spectrogram.shape}")
            logger.info(f"Loaded mask shape: {mask_data.shape}")

            # Ensure mask has shape (H, W) for consistency
            if mask_data.ndim == 3:
                self.mask = mask_data.squeeze(axis=-1).copy()
            else:
                self.mask = mask_data.copy()

            # Validate that spectrogram and mask have matching dimensions
            if self.spectrogram.shape != self.mask.shape:
                raise ValueError(
                    f"Shape mismatch: spectrogram {self.spectrogram.shape} "
                    f"vs mask {self.mask.shape}"
                )

            self.mask_modified = False

            # Reset undo stack and stroke preview for new data
            self.undo_stack = []
            self.stroke_preview_mask = None

            # Create cached base image (spectrogram without mask)
            self.create_base_image()

            # Render the canvas
            self.render_canvas()
            self.render_minimap()

            # Update edited status
            self.update_edited_status()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def _navigate(self, direction: int):
        """
        Navigate to previous/next channel or shot.

        Args:
            direction: -1 for previous, 1 for next
        """
        if self.current_shot is None:
            return

        channels = self.get_available_channels(self.current_shot)
        current_channel_idx = (
            channels.index(self.current_channel)
            if self.current_channel in channels
            else 0
        )

        # Try to navigate within current shot's channels
        new_channel_idx = current_channel_idx + direction

        if 0 <= new_channel_idx < len(channels):
            # Navigate within current shot
            self.channel_combo.current(new_channel_idx)
        else:
            # Navigate to different shot
            current_shot_idx = self.shot_numbers.index(self.current_shot)
            new_shot_idx = current_shot_idx + direction

            if 0 <= new_shot_idx < len(self.shot_numbers):
                self.shot_combo.current(new_shot_idx)
                self.update_channel_list()
                # Select first or last channel depending on direction
                new_shot_channels = self.get_available_channels(
                    self.shot_numbers[new_shot_idx]
                )
                if new_shot_channels:
                    channel_idx = 0 if direction > 0 else len(new_shot_channels) - 1
                    self.channel_combo.current(channel_idx)

        self.load_data()

    def load_previous(self):
        """Load the previous channel or shot."""
        self._navigate(-1)

    def load_next(self):
        """Load the next channel or shot."""
        self._navigate(1)

    def create_base_image(self):
        """Create and cache the base spectrogram image (without mask overlay)."""
        if self.spectrogram is None:
            return

        # Clip and normalize spectrogram (only done once!)
        img_data = self.clip_image(self.spectrogram)  # (F, W)

        # Convert to RGB
        img_rgb = np.stack([img_data, img_data, img_data], axis=-1)  # (F, W, 3)

        # Flip vertically to correct orientation
        img_rgb = np.flipud(img_rgb)

        # Store as cached base image
        self.base_img_rgb = img_rgb

    def _pil_to_photoimage(self, pil_image: Image.Image) -> tk.PhotoImage:
        """
        Convert PIL Image to tkinter PhotoImage using PPM buffer workaround.

        Args:
            pil_image: PIL Image to convert

        Returns:
            tkinter PhotoImage
        """
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PPM")
        buffer.seek(0)
        return tk.PhotoImage(data=buffer.getvalue())

    def clip_image(self, data: np.ndarray) -> np.ndarray:
        """
        Clip spectrogram data to percentiles and normalize to 0-255.

        Args:
            data: Spectrogram data with shape (F, W) - already log-magnitude

        Returns:
            Normalized image array (F, W) with values 0-255
        """
        # Clip to percentiles
        low = np.percentile(data, self.settings["percentile_low"])
        high = np.percentile(data, self.settings["percentile_high"])

        clipped = np.clip(data, low, high)

        # Normalize to 0-255
        return ((clipped - low) / (high - low) * 255).astype(np.uint8)

    def _apply_mask_overlay(self, base_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask overlay to base image.

        Args:
            base_img: Base RGB image (H, W, 3), already flipped
            mask: Binary mask (H, W), not flipped

        Returns:
            RGB image with mask overlay applied
        """
        # Flip mask to match flipped base image
        mask_binary = np.flipud(mask > 0)

        # Create mask overlay using vectorized numpy operations
        mask_r, mask_g, mask_b = self.settings["mask_color"]
        alpha = self.settings["mask_alpha"]
        mask_color = np.array([mask_r, mask_g, mask_b], dtype=np.uint8)

        # Vectorized overlay: blend base image with mask color where mask is True
        return np.where(
            mask_binary[:, :, np.newaxis],
            ((1 - alpha) * base_img + alpha * mask_color).astype(np.uint8),
            base_img,
        )

    def render_stroke_preview(self):
        """
        Render only the visible viewport with stroke preview during drawing.
        This avoids regenerating the entire large image.
        """
        if self.base_img_rgb is None or self.mask is None or self.display_image is None:
            return

        # Don't update if stroke preview is empty
        if self.stroke_preview_mask is None or not np.any(self.stroke_preview_mask):
            return

        # Get visible viewport region
        x1_frac, x2_frac = self.canvas.xview()
        y1_frac, y2_frac = self.canvas.yview()

        img_width = self.display_image.width
        img_height = self.display_image.height

        # Calculate pixel coordinates of visible region
        x1_px = int(x1_frac * img_width)
        x2_px = int(x2_frac * img_width)
        y1_px = int(y1_frac * img_height)
        y2_px = int(y2_frac * img_height)

        # Add some padding for smoother experience
        padding = 100
        x1_px = max(0, x1_px - padding)
        x2_px = min(img_width, x2_px + padding)
        y1_px = max(0, y1_px - padding)
        y2_px = min(img_height, y2_px + padding)

        # Extract only the visible region
        viewport_base = self.base_img_rgb[y1_px:y2_px, x1_px:x2_px].copy()
        viewport_mask = self.mask[y1_px:y2_px, x1_px:x2_px]
        viewport_preview = self.stroke_preview_mask[y1_px:y2_px, x1_px:x2_px]

        # Apply mask overlay to viewport
        img_rgb = self._apply_mask_overlay(viewport_base, viewport_mask)

        # Add stroke preview in a different color (cyan/blue)
        preview_binary = np.flipud(viewport_preview)
        preview_color = np.array([0, 255, 255], dtype=np.uint8)  # Cyan
        preview_alpha = 0.6
        img_rgb = np.where(
            preview_binary[:, :, np.newaxis],
            ((1 - preview_alpha) * img_rgb + preview_alpha * preview_color).astype(
                np.uint8
            ),
            img_rgb,
        )

        # Create PIL Image for viewport
        viewport_image = Image.fromarray(img_rgb, mode="RGB")
        viewport_photo = self._pil_to_photoimage(viewport_image)

        # Update only the viewport region on canvas
        self.canvas.delete("viewport_preview")
        self.canvas.create_image(
            x1_px, y1_px, anchor=tk.NW, image=viewport_photo, tags="viewport_preview"
        )

        # Keep reference to prevent garbage collection
        self._viewport_preview_photo = viewport_photo

    def render_cursor_circle(self):
        """Render brush cursor circle as canvas overlay (no image regeneration)."""
        if self.cursor_position is None:
            # Remove cursor circle if it exists
            if self.cursor_circle_item is not None:
                self.canvas.delete(self.cursor_circle_item)
                self.cursor_circle_item = None
            return

        cx, cy = self.cursor_position
        radius = self.brush_size.get() // 2

        # Remove old cursor circle
        if self.cursor_circle_item is not None:
            self.canvas.delete(self.cursor_circle_item)

        # Draw new cursor circle as canvas item
        self.cursor_circle_item = self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="yellow",
            width=2,
        )

    def render_canvas(self):
        """Render the spectrogram with mask overlay on the canvas."""
        if self.base_img_rgb is None or self.mask is None:
            return

        # Apply mask overlay to cached base image
        img_rgb = self._apply_mask_overlay(self.base_img_rgb.copy(), self.mask)

        # Create PIL Image
        self.display_image = Image.fromarray(img_rgb, mode="RGB")

        # Create PhotoImage for canvas
        self.photo_image = self._pil_to_photoimage(self.display_image)

        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Update canvas scroll region
        img_width = self.display_image.width
        img_height = self.display_image.height
        self.canvas.config(scrollregion=(0, 0, img_width, img_height))

        # Update canvas size to match image height
        self.canvas.config(height=min(img_height, self.canvas_height))

    def update_edited_status(self):
        """Update the edited status label to show if current file has been edited."""
        if self.current_shot is None or self.current_channel is None:
            return

        # Check if edited file exists in output directory
        output_path = (
            self.settings["output_dir"]
            / f"{self.current_shot}_{self.current_channel}_mask.joblib"
        )

        if output_path.exists():
            self.edited_status_label.config(
                text="✓ Previously edited", foreground="green"
            )
        else:
            self.edited_status_label.config(text="Not edited yet", foreground="gray")

    def render_minimap(self):
        """Render a minimap showing the full spectrogram and viewport position."""
        if self.display_image is None:
            return

        # Calculate thumbnail size - prioritize filling height for wide spectrograms
        img_width = self.display_image.width
        img_height = self.display_image.height

        # Scale to fill height, crop width if necessary
        scale = self.MINIMAP_HEIGHT / img_height
        thumb_width = int(img_width * scale)
        thumb_height = int(img_height * scale)

        # If width exceeds minimap width, limit it
        if thumb_width > self.MINIMAP_WIDTH:
            scale = self.MINIMAP_WIDTH / img_width
            thumb_width = self.MINIMAP_WIDTH
            thumb_height = int(img_height * scale)

        thumbnail = self.display_image.resize(
            (thumb_width, thumb_height), Image.Resampling.LANCZOS
        )

        # Convert thumbnail to PhotoImage
        self.minimap_photo = self._pil_to_photoimage(thumbnail)

        # Clear and draw minimap
        self.minimap_canvas.delete("all")
        self.minimap_canvas.create_image(
            self.MINIMAP_WIDTH // 2,
            self.MINIMAP_HEIGHT // 2,
            anchor=tk.CENTER,
            image=self.minimap_photo,
        )

        # Draw viewport indicator
        # This will be updated when scrolling
        self.update_minimap_viewport()

    def update_minimap_viewport(self):
        """Update the viewport indicator on the minimap."""
        if self.display_image is None:
            return

        # Get current viewport position
        x1_frac, x2_frac = self.canvas.xview()

        # Calculate viewport rectangle on minimap using same logic as render_minimap
        img_width = self.display_image.width
        img_height = self.display_image.height

        # Scale to fill height first
        scale = self.MINIMAP_HEIGHT / img_height
        thumb_width = int(img_width * scale)
        thumb_height = int(img_height * scale)

        # If width exceeds minimap width, limit it
        if thumb_width > self.MINIMAP_WIDTH:
            scale = self.MINIMAP_WIDTH / img_width
            thumb_width = self.MINIMAP_WIDTH
            thumb_height = int(img_height * scale)

        # Center offset
        offset_x = (self.MINIMAP_WIDTH - thumb_width) // 2
        offset_y = (self.MINIMAP_HEIGHT - thumb_height) // 2

        # Viewport rectangle coordinates
        vp_x1 = offset_x + x1_frac * thumb_width
        vp_x2 = offset_x + x2_frac * thumb_width
        vp_y1 = offset_y
        vp_y2 = offset_y + thumb_height

        # Remove old viewport indicator
        if self.viewport_rect is not None:
            self.minimap_canvas.delete(self.viewport_rect)

        # Draw new viewport indicator
        self.viewport_rect = self.minimap_canvas.create_rectangle(
            vp_x1, vp_y1, vp_x2, vp_y2, outline="yellow", width=2
        )

    def on_mouse_press(self, event):
        """Handle mouse press event."""
        if self.mask is None:
            return

        # Save state for undo before starting new stroke
        self._save_undo_state()

        self.drawing = True
        self.current_stroke_modified = False  # Reset for new stroke
        self.last_x = self.canvas.canvasx(event.x)
        self.last_y = self.canvas.canvasy(event.y)

        # Initialize stroke preview mask
        self.stroke_preview_mask = np.zeros_like(self.mask, dtype=bool)

        # Draw at initial position
        self.draw_at_position(self.last_x, self.last_y)

    def on_mouse_drag(self, event):
        """Handle mouse drag event."""
        if (
            not self.drawing
            or self.mask is None
            or self.last_x is None
            or self.last_y is None
        ):
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Draw line from last position to current
        self.draw_line(self.last_x, self.last_y, x, y)

        self.last_x = x
        self.last_y = y

    def on_mouse_release(self, event):
        """Handle mouse release event."""
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self._frame_counter = 0  # Reset frame counter

        # Clear stroke preview
        self.stroke_preview_mask = None

        # Remove viewport preview overlay from canvas
        self.canvas.delete("viewport_preview")
        self._viewport_preview_photo = None

        # Render the full canvas after drawing is complete (only if something changed)
        if self.current_stroke_modified:
            self.render_canvas()
            self.render_minimap()
            self.current_stroke_modified = False

    def on_mouse_move(self, event):
        """Handle mouse move event for cursor preview."""
        if self.mask is None:
            return

        # Update cursor position
        self.cursor_position = (
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )

        # If not drawing, render cursor circle (lightweight canvas overlay)
        if not self.drawing:
            self.render_cursor_circle()

    def on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        self.cursor_position = None
        # Just remove the cursor circle, no need to re-render
        self.render_cursor_circle()

    def draw_at_position(self, x: float, y: float):
        """Draw or erase at the given position."""
        if self.mask is None:
            return

        # Convert canvas coordinates to image coordinates
        img_x = int(x)
        # Flip y coordinate since we flipped the image vertically (mask is 2D: H, W)
        img_y = self.mask.shape[0] - 1 - int(y)

        # Get brush parameters
        brush_size = self.brush_size.get()
        mode = self.draw_mode.get()

        # Draw circular brush on mask
        self.apply_brush(img_x, img_y, brush_size, mode)

        # Also update stroke preview mask
        if self.stroke_preview_mask is not None:
            self.apply_brush_to_preview(img_x, img_y, brush_size, mode)

        # Mark as modified
        self.mask_modified = True
        self.current_stroke_modified = True

        # Render with stroke preview during drawing (with frame skipping for performance)
        if self.drawing:
            self._frame_counter += 1
            if self._frame_counter >= self._frame_skip:
                self.render_stroke_preview()
                self._frame_counter = 0

    def draw_line(self, x1: float, y1: float, x2: float, y2: float):
        """Draw or erase a line between two points."""
        # Interpolate points along the line
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_points = max(int(distance), 1)

        for i in range(num_points + 1):
            t = i / max(num_points, 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            self.draw_at_position(x, y)

    def apply_brush(self, x: int, y: int, brush_size: int, mode: str):
        """Apply brush to the mask at the given position using vectorized operations."""
        if self.mask is None:
            return
        H, W = self.mask.shape

        # Create circular brush
        radius = brush_size // 2

        # Determine the value to set
        value = 1.0 if mode == "draw" else 0.0

        # Calculate bounds
        y_min = max(0, y - radius)
        y_max = min(H, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(W, x + radius + 1)

        # Create coordinate grids for the brush region
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

        # Create circular mask using distance from center
        circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

        # Apply brush value where circle mask is True
        self.mask[y_min:y_max, x_min:x_max][circle_mask] = value

    def apply_brush_to_preview(self, x: int, y: int, brush_size: int, mode: str):
        """Apply brush to the preview mask to show current stroke using vectorized operations."""
        if self.stroke_preview_mask is None:
            return
        H, W = self.stroke_preview_mask.shape

        # Create circular brush
        radius = brush_size // 2

        # Calculate bounds
        y_min = max(0, y - radius)
        y_max = min(H, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(W, x + radius + 1)

        # Create coordinate grids for the brush region
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

        # Create circular mask using distance from center
        circle_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

        # Mark as True in preview where circle mask is True
        self.stroke_preview_mask[y_min:y_max, x_min:x_max][circle_mask] = True

    def _save_undo_state(self):
        """Save current mask state to undo stack."""
        if self.mask is not None:
            # Add current state to undo stack
            self.undo_stack.append(self.mask.copy())
            # Limit undo stack size
            if len(self.undo_stack) > self.max_undo_levels:
                self.undo_stack.pop(0)

    def undo(self):
        """Undo the last mask modification."""
        if not self.undo_stack:
            return

        # Restore previous state
        self.mask = self.undo_stack.pop()
        self.mask_modified = True

        # Re-render
        self.render_canvas()
        self.render_minimap()

    def _adjust_brush_size(self, delta: int):
        """
        Adjust brush size by delta.

        Args:
            delta: Amount to change brush size (positive or negative)
        """
        current = self.brush_size.get()
        new_size = max(1, min(50, current + delta))  # Clamp between 1 and 50
        self.brush_size.set(new_size)

    def restore_original(self):
        """Restore the original mask, discarding any edits."""
        if self.current_shot is None or self.current_channel is None:
            messagebox.showwarning("Warning", "No data loaded")
            return

        # Check if edited version exists
        output_path = (
            self.settings["output_dir"]
            / f"{self.current_shot}_{self.current_channel}_mask.joblib"
        )

        if not output_path.exists():
            messagebox.showinfo(
                "Info", "No edited version exists - already showing original"
            )
            return

        # Confirm with user
        response = messagebox.askyesno(
            "Restore Original",
            "This will discard all edits and restore the original mask. Continue?",
        )

        if response:
            # Reload from input directory (original)
            self.load_data(source="input")
            logger.info("Restored original mask")

    def save_mask(self):
        """Save the modified mask to the output directory."""
        if self.mask is None or self.current_shot is None:
            messagebox.showwarning("Warning", "No data loaded to save")
            return

        try:
            # Save to output directory as 2D array (H, W)
            output_path = (
                self.settings["output_dir"]
                / f"{self.current_shot}_{self.current_channel}_mask.joblib"
            )
            joblib.dump(self.mask, output_path, compress=True)

            self.mask_modified = False
            logger.info(f"Saved mask to {output_path}")
            messagebox.showinfo("Success", f"Mask saved to {output_path.name}")

            # Update edited status
            self.update_edited_status()

        except Exception as e:
            logger.error(f"Error saving mask: {e}")
            messagebox.showerror("Error", f"Failed to save mask: {e}")

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main(config_path: Path | str | None = None) -> None:
    """Main function to launch the GUI."""
    # Configure logging to display to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    settings = load_settings(config_path, default_settings)

    # Create and run the editor
    editor = SpectrogramEditor(settings)
    editor.run()


if __name__ == "__main__":
    # python -m autotslabel.autosegment.multichannel.step_5b_manual_fix_spectrogram
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
