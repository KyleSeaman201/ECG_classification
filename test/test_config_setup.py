import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from config.config_setup import LocalVariables

config_file = project_root.joinpath("config").joinpath("configuration.yaml")
print(f"config_file {config_file}")
local_var = LocalVariables(yaml_config_file=config_file)

print(f"data root: {local_var.data_root}")
print(f"dataloader workers: {local_var.dataloader_workers}")
print(f"mit bih data: {local_var.mitbih_data}")
print(f"mit bih train: {local_var.mitbih_train}")
print(f"mit bih test: {local_var.mitbih_test}")
print(f"ptbdb data: {local_var.ptbdb_data}")
print(f"ptbdb abnormal: {local_var.ptbdb_abnormal}")
print(f"ptbdb normal: {local_var.ptbdb_normal}")
print(f"one d cnn info: {local_var.one_d_cnn_info}")
print(f"one d cnn optimizer info: {local_var.one_d_cnn_optim}")
print(f"two d cnn info: {local_var.two_d_cnn_info}")
print(f"hybrid info: {local_var.hybrid_info}")
print(f"num head: {local_var.n_head}")
print(f"num layers: {local_var.n_layers}")
print(f"num conv: {local_var.n_conv}")
print(f"num filters: {local_var.n_filters}")
print(f"model hidden units: {local_var.d_hid}")
print(f"model dimension: {local_var.d_model}")
print(f"model projection: {local_var.d_proj}")
print(f"pyspark info: {local_var.pyspark_info}")
print(f"parquet info: {local_var.parquet_dir}")
print(f"parquet files: {local_var.parquet_files}")
print(f"mitbih class map: {local_var.mitbih_class_map}")
print(f"ptbdb class map: {local_var.ptbdb_class_map}")
