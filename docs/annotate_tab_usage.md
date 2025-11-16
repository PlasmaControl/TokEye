# Annotate Tab - User Guide

## Quick Start

### Step 1: Prepare Your Data
Ensure you have .npy files containing your spectrograms or transformed data:
```
data/
├── spectrogram_001.npy
├── spectrogram_002.npy
└── spectrogram_003.npy
```

### Step 2: Load a File
1. Click "Upload .npy File" in the left panel
2. Select your spectrogram file
3. Click "Load for Annotation"

The system will:
- Display your spectrogram as a backdrop
- Check for existing annotations
- Load existing mask OR create an empty one

### Step 3: Annotate
Use the annotation canvas to mark regions:

**Drawing:**
- Select the red brush (default)
- Click and drag to mark regions
- Adjust brush size with the slider

**Erasing:**
- Select the eraser tool
- Click and drag to remove annotations
- Adjust eraser size as needed

**Zooming:**
- Use mouse wheel to zoom in/out
- Precise annotations at high zoom levels

### Step 4: Preview (Optional)
To see your mask without the backdrop:
1. Open the "Mask Preview" accordion
2. Click "Update Preview"
3. View:
   - **Mask Only**: Your annotations (black/white)
   - **Backdrop Only**: Reference image

### Step 5: Save Your Work
1. Choose save format:
   - **npy**: Binary numpy array (recommended)
   - **png**: Grayscale image (for visualization)
2. Click "Save Mask"
3. Confirmation will show: `annotations/{filename}_mask.{ext}`

## Key Concepts

### Backdrop vs. Mask

**Backdrop:**
- Your original .npy file (spectrogram, transform output)
- Read-only, never modified
- Displayed as reference only

**Mask:**
- Your annotations overlay
- Editable, saved separately
- Located in `annotations/` directory

### File Naming

**Input:** `spectrogram_20240115.npy`
**Output:** `annotations/spectrogram_20240115_mask.npy`

The mask is always named after the original file for easy association.

### Editing Existing Annotations

When you reload a file that already has annotations:
1. Upload the same .npy file
2. Click "Load for Annotation"
3. System automatically finds and loads your existing mask
4. Continue editing where you left off
5. Save to overwrite the existing mask

## Tips and Best Practices

### Annotation Strategy

1. **Start with low zoom** for broad regions
2. **Zoom in** for precise boundaries
3. **Use consistent colors** (red for primary annotations)
4. **Save frequently** to avoid data loss

### Brush Selection

- **Red brush**: Primary annotations (regions of interest)
- **Green brush**: Alternative/secondary regions
- **Blue brush**: Different classification
- **White brush**: Highlighting

### Workflow Optimization

**Single file:**
```
1. Load → 2. Annotate → 3. Save
```

**Multiple files:**
```
1. Load file A → 2. Annotate → 3. Save
4. Load file B → 5. Annotate → 6. Save
7. Load file C → 8. Annotate → 9. Save
```

**Refinement:**
```
1. Load previously annotated file
2. System loads existing mask
3. Make corrections
4. Save (overwrites previous)
```

### Quality Control

Before saving, check:
- ✓ All target regions marked
- ✓ No stray annotations
- ✓ Boundaries are precise
- ✓ Preview looks correct

## Common Workflows

### Marking Plasma Events

```
Goal: Annotate disruption events in tokamak spectrograms

1. Load spectrogram_shot_12345.npy
2. Backdrop shows time-frequency spectrogram
3. Use red brush to mark disruption regions
4. Zoom in for precise temporal boundaries
5. Save as annotations/spectrogram_shot_12345_mask.npy
```

### Creating Training Data

```
Goal: Generate labeled dataset for ML model

For each spectrogram:
1. Load file
2. Annotate regions (e.g., MHD activity)
3. Save mask
4. Repeat for all files

Result:
data/
├── spectrogram_001.npy    →    annotations/spectrogram_001_mask.npy
├── spectrogram_002.npy    →    annotations/spectrogram_002_mask.npy
└── spectrogram_003.npy    →    annotations/spectrogram_003_mask.npy
```

### Correcting Model Predictions

```
Goal: Refine automated predictions

1. Load spectrogram with existing prediction
2. System loads predicted mask
3. Correct false positives (use eraser)
4. Add false negatives (use brush)
5. Save corrected mask
```

## File Organization

### Recommended Structure

```
project/
├── data/                      # Original spectrogram data
│   ├── shot_001.npy
│   ├── shot_002.npy
│   └── shot_003.npy
├── annotations/               # Manually annotated masks
│   ├── shot_001_mask.npy
│   ├── shot_002_mask.npy
│   └── shot_003_mask.npy
└── exports/                   # Optional PNG exports
    ├── shot_001_mask.png
    ├── shot_002_mask.png
    └── shot_003_mask.png
```

### Backup Strategy

1. **Version control** the `annotations/` directory
2. **Keep original data** in a separate location
3. **Export to PNG** for visual records
4. **Document** annotation conventions

## Troubleshooting

### Issue: "No file loaded"
**Solution:** Ensure you've clicked "Load for Annotation" after uploading

### Issue: "Could not extract mask"
**Solution:**
- Ensure you've drawn annotations before saving
- Check that canvas has content

### Issue: Mask not loading on second attempt
**Solution:**
- Verify mask file exists in `annotations/`
- Check filename matches pattern: `{basename}_mask.npy`
- Ensure mask shape matches backdrop shape

### Issue: Annotations look wrong after reload
**Solution:**
- Check that you saved after last edit
- Verify correct file was loaded
- Try "Update Preview" to see mask separately

### Issue: Save failed
**Solution:**
- Check write permissions on `annotations/` directory
- Ensure sufficient disk space
- Verify filename is valid (no special characters)

## Advanced Features

### Mask Preview

Use the preview to:
- **Verify coverage**: Ensure all regions marked
- **Check precision**: View mask boundaries
- **Compare to backdrop**: See overlay separately

### Format Selection

**NPY format (.npy):**
- ✓ Lossless binary storage
- ✓ Fast loading/saving
- ✓ Preserves exact pixel values
- ✓ Compatible with numpy workflows

**PNG format (.png):**
- ✓ Visual inspection
- ✓ Shareable with others
- ✓ Compatible with image viewers
- ⚠ May lose precision (8-bit grayscale)

**Recommendation:** Save as .npy for primary storage, export as .png for sharing

## Keyboard Shortcuts

(Note: Dependent on Gradio ImageEditor implementation)

- **Mouse wheel**: Zoom in/out
- **Click + Drag**: Draw/erase
- **Right click**: Context menu (if available)

## Integration with TokEye Pipeline

### Typical Workflow

```
1. Load signal data
2. Apply transforms (spectrogram, wavelet, etc.)
3. Export transform as .npy
4. Annotate in Annotate Tab
5. Use annotations for:
   - Training ML models
   - Validating predictions
   - Creating labeled datasets
   - Research analysis
```

### Data Flow

```
Raw Signal → Transform → .npy Export
                           ↓
                    Annotate Tab
                           ↓
                    Mask Annotation
                           ↓
                    Save to annotations/
                           ↓
                    Use in ML Pipeline
```

## Best Practices Summary

1. **Always save your work** - Click save after each annotation session
2. **Use consistent naming** - Let the system handle filename conventions
3. **Preview before saving** - Use mask preview to verify
4. **Backup annotations** - Version control or regular backups
5. **Document conventions** - Keep notes on what colors mean
6. **Quality over speed** - Precise annotations are worth the time
7. **Start simple** - Mark obvious regions first, refine later
8. **Use zoom** - For precise boundary marking

## Getting Help

If you encounter issues not covered here:

1. Check the main documentation: `ANNOTATE_TAB_REDESIGN.md`
2. Run tests to verify installation: `uv run python test_annotate.py`
3. Review function docstrings in `src/TokEye/gradio/tabs/annotate.py`
4. Check console for error messages

## Example Session

```
Session: Annotating Shot #12345 Spectrogram

1. [09:00] Load spectrogram_shot_12345.npy
   - Backdrop displays: Time-frequency spectrogram
   - Info: "No existing mask found - created empty mask"

2. [09:05] Annotate disruption event
   - Use red brush, size 5
   - Mark region around t=2.5s, f=100-200kHz
   - Zoom in for precise boundaries

3. [09:10] Preview annotation
   - Click "Update Preview"
   - Verify: Disruption region clearly marked

4. [09:12] Save annotation
   - Format: npy
   - Click "Save Mask"
   - Confirmation: "Mask saved to annotations/spectrogram_shot_12345_mask.npy"

5. [09:15] Continue to next shot
   - Load spectrogram_shot_12346.npy
   - Repeat process

Session complete: 2 files annotated in 15 minutes
```
