import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read, write
from src.graph import KNNGraph
from src.module import Module
from src.transforms import LineGraph, RandomPerturbation

torch.set_float32_matmul_precision("high")
torch.serialization.add_safe_globals([LineGraph, RandomPerturbation()])


def run_standalone_inference(checkpoint_path, input_file, metadata_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct Label Mapping from metadata (original space group numbers)
    df_meta = pd.read_csv(metadata_csv)
    unique_labels = sorted(set(df_meta["SpaceGroupNumber"]))
    index_to_label = {idx: int(label) for idx, label in enumerate(unique_labels)}
    print("Label mapping reconstructed.")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    k = ckpt["datamodule_hyper_parameters"]["k"]

    # This removes the '_orig_mod.' prefix added by torch.compile
    raw_state_dict = ckpt["state_dict"]
    clean_state_dict = {}
    for key, val in raw_state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        clean_state_dict[new_key] = val

    ckpt["hyper_parameters"]["compile"] = False
    model = Module(**ckpt["hyper_parameters"])
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded for inference.")

    # 3. Builders / transforms
    knn_builder = KNNGraph(k=k)
    lg_transform = LineGraph()

    atoms: Atoms = read(input_file)
    atoms.rattle(0.01)  # Small random perturbation to avoid degenerate positions

    if Path("graph.pckl").exists():
        lg_full = torch.load("graph.pckl", weights_only=False)
    else:
        graph = knn_builder.convert(atoms_repr=atoms)
        lg_full = lg_transform(graph)
    print("Graph constructed from input structure.")

    with torch.inference_mode():
        start = time.perf_counter()
        logits = model(lg_full.to(device))
        end = time.perf_counter()
        preds = logits.argmax(dim=-1).cpu()
        all_preds = [index_to_label[pred.item()] for pred in preds]
    print(f"Inference time: {end - start:.4f} seconds")

    # Attach predictions and write output
    atoms.set_array("prediction", np.array(all_preds))
    output_path = Path(input_file).stem + "_predicted.xyz"
    write(output_path, atoms, format="extxyz")

    print(f"Inference complete. Output saved to {output_path}")


if __name__ == "__main__":
    run_standalone_inference(
        checkpoint_path="best_model.ckpt",
        input_file="POSCAR",
        metadata_csv="data/csg/raw/CSG_tiny.csv",
    )
