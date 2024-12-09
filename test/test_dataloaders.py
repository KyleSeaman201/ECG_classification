import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from data.dataloaders import pre_process_data, get_dataloader
from config.config_setup import LocalVariables
from data.data_transformation import DataTransformation

def main(local_config):
    # pre process the data frames by generating parquet files for the various datasets
    pre_process_data(local_config)

    for k, pf in local_config.parquet_files.items():
        
        # set something small to just check one iteration
        batch_size = 16

        # check for transformation first
        dataTransformation = DataTransformation()
        transform = transforms.Lambda(dataTransformation.transform)
        dataloader = get_dataloader(pf, batch_size, local_config.dataloader_workers, transform=transform)

        one_iteration = next(iter(dataloader))
        print(f"Testing for {pf} using transformation")
        print(f"one iteration shape for first batch features: {one_iteration[0].shape}")
        print(f"one iteration shape for labels: {one_iteration[1].shape}")

        if "train" in pf:
            train_dataloader, valid_dataloader = get_dataloader(pf, batch_size, local_config.dataloader_workers, split=True)
            one_iteration = next(iter(train_dataloader))
            print(f"Testing for {pf} using split of data")
            print(f"one iteration of train shape for first batch features: {one_iteration[0].shape}")
            print(f"one iteration of train shape for labels: {one_iteration[1].shape}")
            one_iteration = next(iter(valid_dataloader))
            print(f"one iteration of valid shape for first batch features: {one_iteration[0].shape}")
            print(f"one iteration of valid shape for labels: {one_iteration[1].shape}")
        
        dataloader = get_dataloader(pf, batch_size, local_config.dataloader_workers)
        one_iteration = next(iter(dataloader))
        print(f"Testing for {pf} without transformation")
        print(f"one iteration shape for first batch features: {one_iteration[0].shape}")
        print(f"one iteration shape for labels: {one_iteration[1].shape}")

    # test the npy file for each of the datasets
    print(f"Testing the NPY files for class weights for each dataset")
    for d in ["mitbih", "ptbdb"]:
        npy_file_name = str(project_root.joinpath("data").joinpath(d).joinpath(d + "_class_weights.npy"))
        class_weights = torch.tensor(np.load(npy_file_name), dtype=torch.float32)
        print(f"class_weights for {d}: {class_weights}")

if __name__ == '__main__':
    config_file = project_root.joinpath("config").joinpath("configuration.yaml")
    print(f"config_file {config_file}")
    local_config = LocalVariables(yaml_config_file=config_file)

    main(local_config)