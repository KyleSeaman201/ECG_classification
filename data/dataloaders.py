import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

#from petastorm import make_batch_reader
from data.spark_utils import SparkInstance, SparkData
#from petastorm.pytorch import DataLoader

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class ParquetDataset(Dataset):
    def __init__(self, parquet_file, features, targets, transform=None, hybrid = False):
        self.data = pd.read_parquet(parquet_file)
        self.features = features
        self.target = targets
        self.transform = transform
        self.hybrid = hybrid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[self.features].iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.data[self.target].iloc[idx], dtype=torch.float32)

        # for the hyrid model, both raw and transformed data are needed
        if self.hybrid:
            if (self.transform == None):
                print("Transform must not be None when hybrid is True. Please set the transform parameter.")
                raise
            transformed_features = self.transform(features)
            return features, transformed_features, target
        else:
            if self.transform:
                features = self.transform(features)
            return features, target

def get_dataloader(data_path, batch_size=16, num_workers=1, transform=None, hybrid=False, split=False, split_ratio=[0.8, 0.2]):

    # feature columns start with feat_c0 up to feat_c186
    feature_str = "feat_c"

    # create a list of features
    features = [feature_str + str(i) for i in range(187)]

    # label
    target = 'label'

    # create dataset
    dataset = ParquetDataset(data_path, features, target, transform, hybrid)

    # if split data into two sets with given ratio then return the two
    if split == True:
        seed = 42
        generator = torch.Generator().manual_seed(seed)
        train_dataset, valid_dataset = random_split(dataset, split_ratio, generator=generator)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=num_workers)
        return train_dataloader, valid_dataloader
    else:
        # get the dataloader
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

def pre_process_data(local_config):
    processed_data = []
    data_name = None
    is_hybrid = False

    # Check if parquet files are already generated
    mitbih_dir = Path(local_config.parquet_dir['mitbih'])
    ptbdb_dir = Path(local_config.parquet_dir['ptbdb'])
    if (mitbih_dir.is_dir() and ptbdb_dir.is_dir()):
        print("Parquet files exists.")
        return

    # create a spark session
    spark_instance = SparkInstance(local_config.pyspark_info)
    spark_session = spark_instance.get_spark_session()

    # create a spark data
    spark_data = SparkData(spark_session, local_config.pyspark_info, \
        local_config.parquet_dir, local_config.parquet_files)

    # check to ensure files exist
    mitbih_train = local_config.mitbih_train
    assert mitbih_train, f"{mitbih_train} does not exist"

    mitbih_test = local_config.mitbih_test
    assert mitbih_test, f"{mitbih_test} does not exist"

    spark_data.generate_df(mitbih_train, class_map=local_config.mitbih_class_map)
    spark_data.generate_df(mitbih_test, class_map=local_config.mitbih_class_map)
    
    ptbdb_normal = local_config.ptbdb_normal
    assert ptbdb_normal, f"{ptbdb_normal} does not exist"

    ptbdb_abnormal = local_config.ptbdb_abnormal
    assert ptbdb_abnormal, f"{ptbdb_abnormal} does not exist"

    spark_data.generate_join_df([ptbdb_abnormal, ptbdb_normal], class_map=local_config.ptbdb_class_map)

    spark_data.save_class_weights("mitbih_train")
    spark_data.save_class_weights("ptbdb_train")

    spark_data.generate_parquet()

    data_info = spark_data.get_spark_data()
    for k, v in data_info.items():
        print(f"File: {k} Number of Records: {v['num_records']}")

    spark_instance.stop_spark_session()
        
    return



