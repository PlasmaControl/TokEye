import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile as tif
from scipy import ndimage
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils


default_settings = {
    'input_path': 'data/.cache/step_1a_alt',
    'output_path': 'data/.cache/step_2_coco',
    'min_area': 10,
}


def setup_coco_data(mode='train'):
    data = {
        'info': {
            'description': f'AE Mode Dataset - {mode}',
            'version': '1.0',
            'year': datetime.now().year,
            'date_created': datetime.now().isoformat(),
        },
        'licenses': [],
        'categories': [
            {
                'id': 1, 
                'name': 'mode', 
                'supercategory': 'acoustic',
            },
        ],
        'images': [],
        'annotations': [],
    }
    return data


def binary_mask_to_rle(binary_mask):
    """Convert a binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [
        int(cmin), 
        int(rmin), 
        int(cmax - cmin + 1), 
        int(rmax - rmin + 1),
    ]


def extract_instances(
    label_mask, 
    min_area=10,
):
    instances = []
    
    labeled_array, num_features = ndimage.label(label_mask)
    
    for instance_id in range(1, num_features + 1):
        mask = (labeled_array == instance_id)
        area = int(mask.sum())
        
        # Skip small instances
        if area < min_area:
            continue
        
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        
        instances.append({
            'mask': mask,
            'bbox': bbox,
            'area': area,
        })
    
    return instances


def create_coco_dataset(settings, mode='train'):
    
    # Directories
    input_dir = Path(settings['input_path'])
    output_dir = Path(settings['output_path'])
    input_img_dir = input_dir / 'input'
    label_dir = input_dir / 'label'
    output_img_dir = output_dir / f'{mode}'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files for this mode
    pattern = f'*_{mode}.tif'
    input_files = sorted(list(input_img_dir.glob(pattern)))
    print(f"Found {len(input_files)} {mode} files")
    
    # COCO format structure
    coco_data = setup_coco_data(mode)
    
    annotation_id = 1
    
    for img_id, input_file in enumerate(tqdm(
        input_files, desc=f'Processing {mode}'), start=1
        ):
        # Load input image
        input_img = tif.imread(input_file)
        label_file = label_dir / input_file.name
        
        if not label_file.exists():
            print(f"Warning: Label file not found: {label_file}")
            continue
        
        label_mask = tif.imread(label_file)
        label_mask = label_mask > 0
        
        height, width = input_img.shape
        
        img_filename = input_file.stem + '.tif'
        img_path = output_img_dir / img_filename
        tif.imwrite(img_path, input_img)
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_filename,
            'width': width,
            'height': height,
        })
        
        instances = extract_instances(label_mask, min_area=settings['min_area'])
        
        for instance in instances:
            rle = binary_mask_to_rle(instance['mask'])
            
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': instance['bbox'],
                'area': instance['area'],
                'segmentation': rle,
                'iscrowd': 0,
            })
            annotation_id += 1
    
    json_path = output_dir / f'annotations_{mode}.json'
    with open(json_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {json_path}")
    
    return coco_data


if __name__ == '__main__':
    # python -m aemodes.pipeline.step_2a_make_coco
    
    settings = default_settings
    
    # Create output directory
    output_path = Path(settings['output_path'])
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create datasets for train and validation
    create_coco_dataset(settings, mode='train')
    create_coco_dataset(settings, mode='valid')
    
    print("Step 2 completed: COCO dataset created")

