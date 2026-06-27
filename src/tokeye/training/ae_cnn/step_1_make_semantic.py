import tifffile as tif

from pathlib import Path
import shutil

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm.auto import tqdm

from aemodes.utils.dataset import load_dataset

default_settings = {
    "mean": 0.0523,
    "std": 0.0654,
    "threshold": 0.2,
    'data_path': 'data/co2_250_detector.pkl',
    'model_path': 'model/big_mode_v1-5.pt',
    'output_path': 'data/.cache/step_1_make_semantic',
}

def make_semantic(model, dataset, settings, mode='train'):

    for idx in tqdm(range(len(dataset))):

        # make mode proposal
        inp = train_dataset[idx]['X'].unsqueeze(1).to(device)
        inp = (inp - settings['mean']) / settings['std']
        with torch.no_grad():
            out = model(inp)
        out = out[0]
        out = torch.sigmoid(out) > settings['threshold']
        out = out.squeeze(1).cpu().numpy()

        # create label map
        ae_true = train_dataset[idx]['y'].numpy().sum(axis=0) > 0
        ae_true = ae_true[None, :]

        # loop across channels
        for i in range(4):
            overlap = out[i][0] * ae_true # 0 is coherent modes w/ big_mode model

            # input and output array
            inp_array = inp.squeeze(1)[i].cpu().numpy()
            out_array = overlap

            # save overlap
            tif.imwrite(
                f"{settings['output_path']}/input/{idx}_{i}_{mode}.tif", 
                inp_array
                )
            tif.imwrite(
                f"{settings['output_path']}/label/{idx}_{i}_{mode}.tif", 
                out_array
                )

if __name__ == '__main__':
    # python -m aemodes.pipeline.step_1_make_semantic

    settings = default_settings

    # create output path
    input_output_path = Path(settings['output_path']) / 'input'
    label_output_path = Path(settings['output_path']) / 'label'
    if input_output_path.exists():
        shutil.rmtree(input_output_path)
    if label_output_path.exists():
        shutil.rmtree(label_output_path)
    input_output_path.mkdir(parents=True, exist_ok=True)
    label_output_path.mkdir(parents=True, exist_ok=True)

    # load model
    model = torch.load(
        settings['model_path'], 
        weights_only=False, 
        map_location=device
    )
    model.eval()
    print("Model loaded")

    # load data
    train_dataset, valid_dataset = load_dataset(
        settings['data_path']
        )
    
    # make semantic
    make_semantic(model, train_dataset, settings, mode='train')
    make_semantic(model, valid_dataset, settings, mode='valid')

    print("Step 1 completed")