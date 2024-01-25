import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.models.components.cegann import CEGANN
from utils import load_dataset, load_settings, prepare_batch_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_args():
    try:
        validation_data = sys.argv[1]
    except IndexError:
        raise IndexError(
            "For prediction, validation data must be provided as first argument"
        )

    try:
        checkpoint_best = sys.argv[2]
    except IndexError:
        raise IndexError(
            "For prediction, the model checkpoint must be provided as second argument"
        )
        
    if not Path(validation_data).is_dir():
        raise FileNotFoundError("Validation data not found")

    if not Path(checkpoint_best).is_file():
        raise FileNotFoundError("Checkpoint not found")

    return validation_data, checkpoint_best


def predict(model, dataset, labels):
    predictions = {}
    for i in range(dataset.size):
        outdata = dataset.collate([dataset[i]])
        label = labels[i]

        x, y = prepare_batch_fn(outdata, device, non_blocking=False)
        predict, embedding = model(x)

        predictions[label] = {
            "class": np.argmax(predict.cpu().detach().numpy(), axis=1).tolist(),
            "embeddings": embedding.cpu().detach().numpy().tolist(),
        }

    return predictions


def main():
    val_data_path, checkpoint_path = validate_args()
    settings = load_settings()
    graphs, labels = load_dataset(val_data_path, settings, return_labels=True, allow_unknown=True)

    model = CEGANN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        edge_expansion_units=settings.h_fea_edge,
        angle_expansion_units=settings.h_fea_angle,
        n_classes=settings.n_classes,
        pooling=settings.pooling,
        embedding=True,
    )
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model"]) #FIXME: keys are missing

    predictions = predict(model, graphs, labels)
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
