import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse

from data.dataloaders import pre_process_data, get_dataloader
from config.config_setup import LocalVariables
from data.data_transformation import DataTransformation

def parse_args():
    parser = argparse.ArgumentParser(f"{__file__}: --dataset \"mitbih|ptbdb\"")
    parser.add_argument("--dataset", nargs='*', type=str, default="ptbdb")

    parsed_args = parser.parse_args()
    return parsed_args

def main(local_config):
    parsed_args = parse_args()

    pre_process_data(local_config)

    data_name = parsed_args.dataset
    train_parquet = local_config.parquet_files[data_name + "_train"]
    #test_parquet = local_config.parquet_files[data_name + "_test"]

    transform = None

    train_dataloader = get_dataloader(train_parquet, 16, local_config.dataloader_workers, transform=transform)
    #test_dataloader = get_dataloader(test_parquet, 16, local_config.dataloader_workers, transform=transform)


    # obtain one sample of batch from dataloader
    train_iteration = next(iter(train_dataloader))

    dataTransformation = DataTransformation()
    frequencies, times, Sxx = dataTransformation.generate_spectrogram(train_iteration[0][0])
    dataTransformation.plot_spectrogram(frequencies, times, Sxx, plot=True) # This shows the spectrogram
    #dataTransformation.get_spectrogram_plot(frequencies, times, Sxx, plot=True) # This shows the spectrogram that the CNN will be using.

    return



if __name__ == '__main__':
    config_file = project_root.joinpath("config").joinpath("configuration.yaml")
    print(f"config_file {config_file}")
    local_config = LocalVariables(yaml_config_file=config_file)

    main(local_config)