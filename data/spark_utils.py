import sys
import os
from os.path import exists
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import findspark

findspark.init()

import pyspark
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.conf import SparkConf
from sklearn.utils import compute_class_weight
import numpy as np
from plots.plots import PlotClassDistribution


class SparkInstance:
    def __init__(self, pyspark_info):

        self.spark_session = pyspark.sql.SparkSession.builder \
            .appName(pyspark_info["app_name"]) \
            .master(pyspark_info["mode"] + "[" + str(pyspark_info["num_cores"]) + "]") \
            .config("spark.executor.memory", pyspark_info["executor_memory"]) \
            .config("spark.cores.max", pyspark_info["num_cores"]) \
            .config("spark.driver.memory", pyspark_info["driver_memory"]) \
            .getOrCreate()
    
    def get_spark_session(self):
        return self.spark_session
    
    def stop_spark_session(self):
        self.spark_session.stop()
        return

class SparkData:
    def __init__(self, spark_session, pyspark_info, parquet_dir, parquet_files):
        self.df_info = {}
        self.spark_session = spark_session
        self.num_cores = pyspark_info["num_cores"]

        # parquet directory information
        self.parquet_dir = parquet_dir
        self.parquet_files = parquet_files

    def save_class_weights(self, file_name):
        dataset = file_name.split('_')[0]

        df = self.df_info[file_name]["df"]
        labels = np.array(df.select("label").rdd.flatMap(lambda x: x).collect())
        # labels = df['label']

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        #print(class_weights)
        file_name = str(project_root.joinpath("data").joinpath(dataset).joinpath(dataset + "_class_weights.npy"))
        #np.save("data/"+dataset+"/"+dataset+"_class_weights.npy", class_weights)
        np.save(file_name, class_weights)
        return class_weights
    
    def prepend_to_columns(self, df, prefix):
        new_columns = [col(c).alias(prefix+c) if i<len(df.columns) - 1 else c for i, c in enumerate(df.columns)]
        return df.select(*new_columns)

    def generate_df(self, file_path, class_map=None):
        print(f"Generating pyspark dataframe for file: {file_path}")
        file_name = file_path.name
        file_name = file_name.split('.')[0]
        df = self.spark_session.read.csv(str(file_path), header=False, inferSchema=True)
        last_column = df.columns[-1]
        y_df = df.select(last_column)
        num_classes = y_df.distinct().count()
        df = df.withColumnRenamed(df.columns[-1], "label")
        df = self.prepend_to_columns(df, "feat")
        df = df.repartition(self.num_cores)
        self.df_info[file_name] = {}
        self.df_info[file_name]["num_classes"] = num_classes
        self.df_info[file_name]["df"] = df

        # plot the distribution of classes
        unique_classes = df.select('label').distinct().rdd.flatMap(lambda x: x).collect()
        class_count_info = {}
        total_count = 0
        for c in unique_classes:
            count = df.filter(col("label") == c).count()
            class_count_info[c] = count
            total_count += count

        self.df_info[file_name]["num_records"] = total_count

        PlotClassDistribution(class_map, class_count_info, file_name)

        # create a dataframe with the reduce version of data
        file_name_reduced = file_name + '_' + 'reduced'

        # obtain the class with min count
        #min_count = min([ v for k, v in class_count_info.items()])
        #min_count = 10000

        second_largest_count = sorted(class_count_info.values(), reverse=True)[1]
        min_count = second_largest_count//2
        print(min_count)

        # create a random number for each group label to extract the
        # minimum count using the row number since need to use Window 
        window_spec = Window.partitionBy('label').orderBy(F.rand())
        df_with_row_num = df.withColumn("row_num", F.row_number().over(window_spec))
        sampled_df = df_with_row_num.filter(col("row_num") <= min_count).drop("row_num")
        
        self.df_info[file_name_reduced] = {}
        self.df_info[file_name_reduced]["num_classes"] = num_classes
        self.df_info[file_name_reduced]["df"] = sampled_df
        self.df_info[file_name_reduced]["num_records"] = sampled_df.count()

        return
    
    def get_spark_data(self):
        return self.df_info
    
    def generate_join_df(self, file_list, split_ratio=[0.8, 0.2], class_map=None):
        df = None
        file_path = sorted(file_list)[0]
        file_name = file_path.name
        file_name = file_name.split('.')[0]
        file_name = file_name.split('_')[0]
        for file in file_list:
            if df is None:
                df = self.spark_session.read.csv(str(file), header=False, inferSchema=True)
            else:
                temp_df = self.spark_session.read.csv(str(file), header=False, inferSchema=True)
                df = df.union(temp_df)
        
        print(f"Generate pyspark dataframe for file: {file_name}")
        num_classes = df.distinct().count()
        last_column = df.columns[-1]
        unique_classes = df.select(last_column).distinct().rdd.flatMap(lambda x: x).collect()

        # stratify the labels across the data to ensure there is
        # no class imbalance
        def stratified_split(df, label):
            class_df = df.filter(df[last_column] == label)
            train_df, test_df = class_df.randomSplit(split_ratio)
            return train_df, test_df
        
        train_dfs = []
        test_dfs = []
        for label in unique_classes:
            train_df, test_df = stratified_split(df, label)
            train_dfs.append(train_df)
            test_dfs.append(test_df)

        # random split the data
        df_train, df_test = df.randomSplit(split_ratio)
        df_train = df_train.withColumnRenamed(df_train.columns[-1], "label")
        df_train = self.prepend_to_columns(df_train, "feat")
        df_train = df_train.repartition(self.num_cores)
        df_test = df_test.withColumnRenamed(df_test.columns[-1], "label")
        df_test = self.prepend_to_columns(df_test, "feat")
        df_test = df_test.repartition(self.num_cores)

        file_name_train = file_name + '_' + 'train'
        file_name_test = file_name + '_' + 'test'
        self.df_info[file_name_train] = {}
        self.df_info[file_name_test] = {}
        last_column = df_train.columns[-1]
        y_df = df_train.select(last_column)
        num_classes = y_df.distinct().count()
        self.df_info[file_name_train]["df"] = df_train
        self.df_info[file_name_train]["num_classes"] = num_classes
        self.df_info[file_name_train]["num_records"] = y_df.count()
        last_column = df_test.columns[-1]
        y_df = df_test.select(last_column)
        num_classes = y_df.distinct().count()
        self.df_info[file_name_test]["df"] = df_test
        self.df_info[file_name_test]["num_classes"] = num_classes
        self.df_info[file_name_test]["num_records"] = y_df.count()

        # plot the distribution of classes
        unique_classes = df_train.select("label").distinct().rdd.flatMap(lambda x: x).collect()
        class_count_info_train = {}
        for c in unique_classes:
            count = df_train.filter(col("label") == c).count()
            class_count_info_train[c] = count

        PlotClassDistribution(class_map, class_count_info_train, file_name_train)

        unique_classes = df_test.select("label").distinct().rdd.flatMap(lambda x: x).collect()
        class_count_info_test = {}
        for c in unique_classes:
            count = df_train.filter(col("label") == c).count()
            class_count_info_test[c] = count

        PlotClassDistribution(class_map, class_count_info_test, file_name_test)
        return

    def generate_parquet(self):
        for k, v in self.df_info.items():
            data_info = k.split('_')
            data_name = data_info[0]
            is_reduced = True if data_info[-1] == "reduced" else False

            parquet_dir = self.parquet_dir[data_name]
            parquet_dir = Path(parquet_dir)
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_file = self.parquet_files[k]
            print(f"Generating parquet file: {parquet_file}")
            v["df"].write.mode("overwrite").parquet(parquet_file)




