import torch
from ultralytics import YOLO


default_settings = {
    'data_path': 'data/.cache/step_3_yolo/data.yaml',
    'model_save_path': 'model/yolo',
    'num_epochs': 50,
    'batch_size': 32,
    'image_size': 640,
    'model_size': 'n',  # n, s, m, l, x
    'num_workers': 2,
}


def train_yolo_detection(settings=None):
    """Train YOLOv8 for object detection."""
    
    if settings is None:
        settings = default_settings
    
    # Load pretrained model
    model = YOLO(f'yolov8{settings["model_size"]}.pt')
    
    # Train
    results = model.train(
        data=settings['data_path'],
        epochs=settings['num_epochs'],
        batch=settings['batch_size'],
        imgsz=settings['image_size'],
        project=settings['model_save_path'],
        name='detect',
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=settings['num_workers'],
        patience=10,  # Early stopping
        save=True,
        plots=True,
    )
    
    print(f"Detection training completed. Model saved to {settings['model_save_path']}/detect")
    return model


def train_yolo_segmentation(settings=None):
    """Train YOLOv8 for instance segmentation."""
    
    if settings is None:
        settings = default_settings
    
    from ultralytics import YOLO
    
    # Load pretrained segmentation model
    model = YOLO(f'yolov8{settings["model_size"]}-seg.pt')
    
    # Train
    results = model.train(
        data=settings['data_path'],
        epochs=settings['num_epochs'],
        batch=settings['batch_size'],
        imgsz=settings['image_size'],
        project=settings['model_save_path'],
        name='segment',
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=1,
        patience=10,  # Early stopping
        save=True,
        plots=True,
    )
    
    print(f"Segmentation training completed. Model saved to {settings['model_save_path']}/segment")
    return model


def train_yolo(settings=None, task='both'):
    """
    Train YOLOv8 models.
    
    Args:
        settings: Training settings dictionary
        task: 'detect', 'segment', or 'both'
    """
    
    if settings is None:
        settings = default_settings
    
    models = {}
    
    if task in ('detect', 'both'):
        print("Training YOLOv8 Detection...")
        models['detect'] = train_yolo_detection(settings)
    
    if task in ('segment', 'both'):
        print("Training YOLOv8 Segmentation...")
        models['segment'] = train_yolo_segmentation(settings)
    
    return models


if __name__ == '__main__':
    # python -m aemodes.models.detection.train_yolo
    train_yolo(task='both')

