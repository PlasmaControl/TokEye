import json
import shutil
import yaml
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from skimage import measure


default_settings = {
    'input_path': 'data/.cache/step_2_coco',
    'output_path': 'data/.cache/step_3_yolo',
}


def rle_to_polygon(rle_mask, img_height, img_width):
    if isinstance(rle_mask, dict):
        binary_mask = mask_utils.decode(rle_mask)
    else:
        binary_mask = rle_mask
    
    contours = measure.find_contours(binary_mask, 0.5)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=len)
    
    if len(largest_contour) > 100:
        indices = np.linspace(0, len(largest_contour) - 1, 50).astype(int)
        largest_contour = largest_contour[indices]
    
    polygon = []
    for point in largest_contour:
        y, x = point
        x_norm = x / img_width
        y_norm = y / img_height
        polygon.extend([x_norm, y_norm])
    
    return polygon


def coco_bbox_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]


def convert_to_yolo(settings, mode='train'):
    
    input_dir = Path(settings['input_path'])
    output_dir = Path(settings['output_path'])
    
    coco_mode = 'valid' if mode == 'val' else mode
    ann_file = input_dir / f'annotations_{coco_mode}.json'
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images_dir = output_dir / 'images' / mode
    labels_dir = output_dir / 'labels' / mode
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    images = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Converting {len(images)} {mode} images to YOLO format")
    
    for img_id, img_info in tqdm(images.items(), desc=f'Converting {mode}'):
        img_width = img_info['width']
        img_height = img_info['height']
        filename = img_info['file_name']
        
        src_img = input_dir / coco_mode / filename
        dst_img = images_dir / filename
        shutil.copy(src_img, dst_img)
        
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        annotations = annotations_by_image.get(img_id, [])
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['category_id'] - 1
                
                bbox = coco_bbox_to_yolo(ann['bbox'], img_width, img_height)
                
                if 'segmentation' in ann and isinstance(ann['segmentation'], dict):
                    polygon = rle_to_polygon(ann['segmentation'], img_height, img_width)
                    
                    if polygon and len(polygon) >= 6:
                        polygon_str = ' '.join(f'{coord:.6f}' for coord in polygon)
                        f.write(f'{class_id} {polygon_str}\n')
                    else:
                        bbox_str = ' '.join(f'{coord:.6f}' for coord in bbox)
                        f.write(f'{class_id} {bbox_str}\n')
                else:
                    bbox_str = ' '.join(f'{coord:.6f}' for coord in bbox)
                    f.write(f'{class_id} {bbox_str}\n')
    
    print(f"Saved {len(images)} images and labels to {output_dir}")


def create_data_yaml(settings):
    
    output_dir = Path(settings['output_path'])
    
    data_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'mode'
        },
        'nc': 1,
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created {yaml_path}")


if __name__ == '__main__':
    # python -m aemodes.pipeline.step_3_make_yolo
    
    settings = default_settings
    
    # Directories
    output_path = Path(settings['output_path'])
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert train and validation sets
    convert_to_yolo(settings, mode='train')
    convert_to_yolo(settings, mode='val')  # YOLO uses 'val' not 'valid'
    
    # Create data.yaml
    create_data_yaml(settings)