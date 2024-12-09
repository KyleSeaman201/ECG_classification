import yaml
from pathlib import Path

class LocalVariables:
    def __init__(self, yaml_config_file="./configuration.yaml"):
        with open(yaml_config_file, 'r') as file:
            config = yaml.safe_load(file)

        # data directory with respect to root code location
        self.data_root = (Path(__file__).parent.parent).joinpath(config["data_dir"]["data_root"])

        # mitbih data
        self.mitbih_data = self.data_root.joinpath(config["data_dir"]["data_mitbih"])
        self.mitbih_train = self.mitbih_data.joinpath(config["data_mitbih_files"]["train"])
        self.mitbih_test = self.mitbih_data.joinpath(config["data_mitbih_files"]["test"])

        # ptbdb data
        self.ptbdb_data = self.data_root.joinpath(config["data_dir"]["data_ptbdb"])
        self.ptbdb_abnormal = self.ptbdb_data.joinpath(config["data_ptbdb_files"]["abnormal"])
        self.ptbdb_normal = self.ptbdb_data.joinpath(config["data_ptbdb_files"]["normal"])
        
        # Models
        self.model_root = (Path(__file__).parent.parent).joinpath(config["model_dir"]["model_root"])
        self.saved_models = self.model_root.joinpath(config["model_dir"]["saved_models"])

        # dataloader number of workers
        self.dataloader_workers = config["data_dir"]["data_workers"]

        # 1d cnn variables
        self.one_d_cnn_info = {
            "learning_rate" : config["1d_cnn"]["learning_rate"],
            "weight_decay" : config["1d_cnn"]["weight_decay"],
            "batch_size" : config["1d_cnn"]["batch_size"],
            "epochs" : config["1d_cnn"]["epochs"],
            "dropout" : config["1d_cnn"]["dropout"]
        }

        self.one_d_cnn_optim = {
            "beta_1": config["1d_cnn_optim"]["beta_1"],
            "beta_2": config["1d_cnn_optim"]["beta_2"],
            "warm_up" : config["1d_cnn_optim"]["warm_up"]
        }

        # 2d cnn variables
        self.two_d_cnn_info = {
            "learning_rate" : config["2d_cnn"]["learning_rate"],
            "weight_decay" : config["2d_cnn"]["weight_decay"],
            "batch_size" : config["2d_cnn"]["batch_size"],
            "epochs" : config["2d_cnn"]["epochs"],
            "dropout" : config["2d_cnn"]["dropout"]
        }
        
        # cnn configuration
        self.n_head = config["n_head"]
        self.n_layers = config["n_layers"]
        self.n_conv = config["n_conv"]
        self.n_filters = config["n_filters"]
        self.d_hid = config["d_hid"]
        self.d_model =  config["d_model"]
        self.d_proj =  config["d_proj"]

        self.num_classes = {
            "mitbih" : config["data_mitbih_files"]["num_classes"],
            "ptbdb" : config["data_ptbdb_files"]["num_classes"]
        }

        self.parquet_dir = {
            "mitbih" : str(self.mitbih_data.joinpath(config["data_dir"]["data_parquet"])),
            "ptbdb" : str(self.ptbdb_data.joinpath(config["data_dir"]["data_parquet"]))
        }

        self.parquet_files = {
            "mitbih_train" : str(Path(self.parquet_dir["mitbih"]).joinpath("mitbih_train.parquet")),
            "mitbih_test" : str(Path(self.parquet_dir["mitbih"]).joinpath("mitbih_test.parquet")),
            "mitbih_train_reduced" : str(Path(self.parquet_dir["mitbih"]).joinpath("mitbih_trained_reduced.parquet")),
            "mitbih_test_reduced" : str(Path(self.parquet_dir["mitbih"]).joinpath("mitbih_test_reduced.parquet")),
            "ptbdb_train" : str(Path(self.parquet_dir["ptbdb"]).joinpath("ptbdb_train.parquet")),
            "ptbdb_test" : str(Path(self.parquet_dir["ptbdb"]).joinpath("ptbdb_test.parquet"))
        }

        self.hybrid_info = {
            # hybrid specif to files
            "hybrid_ptbdb_lr" : config["hybrid_ptbdb"]["learning_rate"],
            "hybrid_mitbih_lr" : config["hybrid_mitbih"]["learning_rate"],

            "weight_decay" : config["hybrid"]["weight_decay"],
            "batch_size" : config["hybrid"]["batch_size"],
            "epochs" : config["hybrid"]["epochs"]
        }

        # pyspark
        self.pyspark_info = {
            "app_name" : config["pyspark"]["app_name"],
            "num_cores" : config["pyspark"]["num_cores"],
            "mode" : config["pyspark"]["mode"],
            "executor_memory" : config["pyspark"]["executor_memory"],
            "driver_memory" : config["pyspark"]["driver_memory"],
            "cache": str(self.data_root.joinpath(config["pyspark"]["spark_conf_cache_dir"]))
        }

        self.mitbih_class_map = {
            "Normal" : float(config["mitbih_class_map"]["N"]),
            "Supraventricular beats" : float(config["mitbih_class_map"]["S"]),
            "Ventricular beats" : float(config["mitbih_class_map"]["V"]),
            "Fusion beats" : float(config["mitbih_class_map"]["F"]),
            "Unclassified beats" : float(config["mitbih_class_map"]["F"])
        }

        self.ptbdb_class_map = {
            "Normal" : float(config["ptbdb_class_map"]["N"]),
            "Abnormal" : float(config["ptbdb_class_map"]["A"])
        }