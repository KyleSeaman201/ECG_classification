# Authors
Kyle Seaman and Elisabeth Gutierrez-Zaheer

## DESCRIPTION

This project attempts to reproduce the "An hybrid CNN-Transformer model based on multi-feature extraction and attention fusion mechanism for cerebral embolid classification" using the following software high level packages:

* Python
* Pyspark
* Pytorch

A download script is provided to obtain the kaggle ECG Heartbeat Categorization Dataset,
https://www.kaggle.com/datasets/shayanfazeli/heartbeat, which includes two types of Electrocardiagram (ECG) signals datasets. The first dataset contains ECG signals labeled as normal, supraventricular beats, ventricular beats, fusion beats, and unclassified beats. The second dataset consists of normal and abnormal ECG signals. The CSV files are processed in a Pyspark infrastructure to take advantage of distributed parallel processing, with the number of workers specified in the configuration file. This processing converts the data into customized parquet files, which are saved for models to access during training, validation, and testing.

A customized dataset infrastructure uses these parquet files to create batches of tensors for the raw signals, as well as batches of tensors for spectrogram representations of each signal. Additionally, the dataset infrastructure provides APIs to analyze the data distribution and generate weights for the CrossEntropy Loss function, accounting for data imbalance. Each model calls the get_dataloader() APIi to load the parquet files and format the data as required by PyTorch for model training. The models are implemented as class objects in PyTorch and Python, with APIs for internal operations and a high-level object class that is instantiated by a script to train each model.

The following is the structure of the code directory:

* config/
    + config_setup.py
    + configuration.yaml
* data/
    + data_transformation.py
    + dataloaders.py
    + spark_utils.py
* models/
    + CNN2D.py
    + one_d_cnn_transformer.py
    + LateFusion.py
    + IntermediateFusion.py
* plots/
    + plots.py
* test/
    + test_config_setup.py
    + test_data_transformation.py
    + test_dataloaders.py
    + test_one_d_cnn.py
    + test_2d_cnn.py
    + test_late_fusion.py
    + test_intermediate_fusion.py
* download_data.py
* environment.yml
* README.md

The config/ directory provides a configuration file infrastructure that defines various parameters needed for each models implementation. For instance, it specifies the names of the final Parquet files generated from the processed CSV files, as well as key model parameters such as dimensionality, learning rate, weight decay, batch size and more. Using a configuration file simplifies the process of updating parameters for experimentation and reduces the need to hardcode values directly in the code.

The data/ directory contains the data infrastructure, which uses PySpark to preprocess the data into parquet files. These files are then utilized by the DataLoaders infrastructure to create batches for training.

The models/ directory contains the model implementations.

The plots/ directory contains a script to generate plots for accuracy, f1 score, MCC and loss.

The test/ directory contains scripts to train the model, test the PySpark and Dataset infrastructure.

The download_data.py is a download script to extract the data from kaggle.

The environment.yml file contains the software packages used to have a consistent environment for both development and testing.

The README.md provides the description and procedures to run the code.

## PROCEDURES

Before starting the steps to run the code, need to ensure have a login on kaggle. In addition, need to generate an API key since the download script will need this information to access the data. The code can be downloaded by creating a zip file from the GitHub depository. Once download the zip file and extract the content, the directory will be the ROOT directory for the code to run under the Fall-2024-BD4H-project directory. Need to change directory to be inside the Fall-2024-BD4h-project and see the CODE structure described previously.

### ENVIRONMENT SETUP

1. Go to the ROOT directory and check the version of conda configured on the system by using:

    **conda --version**

2. If conda is not configured, please configure conda using https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
3. If conda is configured, please update using the following command:

    **conda update conda**

4. Setup the environment by running the following:

    **conda env create -f environment.yml**

5. The **bd4h_project** environment will be created. Please activate the environment by typing the following:

    **conda activate bd4h_project**

### PRE-PROCESS DATA

1. In the **download_data.py** file, update kaggle_username and kaggle_key to contain your information to use the kaggle api.
2. run the download_data.py as follows:

    **python3 download_data.py**

3. In the ROOT/data directory, the following files should be found:

    - **mitbih/mitbih_train.csv**
    - **mitbih/mitbih_test.csv**
    - **ptbdb/ptbdb_train.csv**
    - **ptbdb/ptbdb_test.csv**

4. If the previous files are not found, please contact support.

5. In the ROOT/test/ directory, run the following command:

    **python3 test_dataloaders.py**

6. The ROOT/data/mitbih/parquet directory, the following parquet files should exist:

    - **mitbih_test.parquet**
    - **mitbih_test_reduced.parquet**
    - **mitbih_train.parquet**
    - **mitbih_train_reduced.parquet**

7. The ROOT/data/ptbdb/parquet directory, the following parquet files should exist:

    - **ptbdb_test.parquet**
    - **ptbdb_train.parquet**

8. The ROOT/data/mitbih directory, the following npy file should exist:

    - **mitbih_class_weights.npy**

9. The ROOT/data/ptbdb directory, the following npy file should exist:

    - **ptbdb_class_weights.npy**

### MODEL TRAINING

#### 1D CNN Transformer

1. In the ROOT/test/ directory, run the following command:

    - **python3 test_one_d_cnn.py --model 1d_cnn_ptbdb --valid**
    - **python3 test_one_d_cnn.py --model 1d_cnn_mitbih --valid --reduce**

The 1D CNN Transformer model struggles in running in the GPU currently available on windows laptop. Attempted to seek a GPU on Google Cloud Computing but none were available. The paper indicated they used a Intel CPU Xeon CPU E5-2650L v3 \@ 1.8 GHz.

2. In the ROOT/plots/images, the following files willl appear:

    - **accuracy_1d_cnn_mitbih.png**
    - **accuracy_1d_cnn_ptbdb.png**
    - **f1_1d_cnn_mitbih.png**
    - **f1_1d_cnn_ptbdb.png**
    - **loss_1d_cnn_ptbdb.png**
    - **loss_1d_cnn_mitbih.png**
    - **mcc_1d_cnn_mitbih.png**
    - **mcc_1d_cnn_ptbdb.png**

These are plots for the various metrics.

3. After a model is trained via the test script above, the model is saved under ROOT/models/saved_models in following directories:

    - **1d_cnn_transformer_mitbih**
    - **1d_cnn_transformer_ptbdb**

    This will be needed to run the fusion layers

4. To test the model using the test dataset, run the following command:

    - **python3 1d_cnn_transformer.py --model 1d_cnn_ptbdb --test**
    - **python3 1d_cnn_trasnformer.py --model 1d_cnn_mitbih --test**

#### 2D CNN

1. In the ROOT/tests/ directory, run the following command:

    - **python3 test_2d_cnn.py --model 2d_cnn_ptbdb --valid**
    - **python3 test_2d_cnn.py --model 2d_cnn_mitbih --valid --reduce**

The paper indicated they used a Intel CPU Xeon CPU E5-2650L v3 \@ 1.8 GHz.

2. In the ROOT/plots/images, the following files willl appear:

    - **accuracy_2d_cnn_mitbih.png**
    - **accuracy_2d_cnn_ptbdb.png**
    - **f1_2d_cnn_mitbih.png**
    - **f1_2d_cnn_ptbdb.png**
    - **loss_2d_cnn_ptbdb.png**
    - **loss_2d_cnn_mitbih.png**
    - **mcc_2d_cnn_mitbih.png**
    - **mcc_2d_cnn_ptbdb.png**

These are plots for the various metrics.

3. After a model is trained via the test script above, the model is saved under ROOT/models/saved_models in following directories:

    - **2d_cnn_mitbih**
    - **2d_cnn_ptbdb**

    This will be needed to run the fusion layers

4. To test the model using the test dataset, run the following command:

    - **python3 test_2d_cnn.py --model 2d_cnn_ptbdb --test**
    - **python3 test_2d_cnn.py --model 2d_cnn_mitbih --test**
#### Late Fusion

1. In the ROOT/tests/ directory, run the following command:

    - **python3 test_late_fusion.py --model late_fusion_ptbdb --valid**
    - **python3 test_late_fusion.py --model late_fusion_mitbih --valid --reduce**

    Note: The saved models for the 1d CNN-Transformer and 2D CNN need to exist prior to running the fusion model.

The paper indicated they used a Intel CPU Xeon CPU E5-2650L v3 \@ 1.8 GHz.

2. In the ROOT/plots/images, the following files willl appear:

    - **accuracy_late_fusion_mitbih.png**
    - **accuracy_late_fusion_ptbdb.png**
    - **f1_late_fusion_mitbih.png**
    - **f1_late_fusion_ptbdb.png**
    - **loss_late_fusion_ptbdb.png**
    - **loss_late_fusion_mitbih.png**
    - **mcc_late_fusion_mitbih.png**
    - **mcc_late_fusion_ptbdb.png**

    These are plots for the various metrics.

3. After a model is trained via the test script above, the model is saved under ROOT/models/saved_models in following directories:

    - **late_fusion_mitbih**
    - **late_fusion_ptbdb**
    
4. To test the model using the test dataset, run the following command:

    - **python3 test_late_fusion.py --model late_fusion_ptbdb --test**
    - **python3 test_late_fusion.py --model late_fusion_mitbih --test**

#### Intermediate Fusion

1. In the ROOT/tests/ directory, run the following command:

    - **python3 test_intermediate_fusion.py --model concat_intermediate_fusion_ptbdb --valid**
    - **python3 test_intermediate_fusion.py --model sum_intermediate_fusion_ptbdb --valid**
    - **python3 test_intermediate_fusion.py --model attention_intermediate_fusion_ptbdb --valid**
    
    - **python3 test_intermediate_fusion.py --model concat_intermediate_fusion_mitbih --valid --reduce**
    - **python3 test_intermediate_fusion.py --model sum_intermediate_fusion_mitbih --valid --reduce**
    - **python3 test_intermediate_fusion.py --model attention_intermediate_fusion_mitbih --valid --reduce**

    Note: The saved models for the 1d CNN-Transformer and 2D CNN need to exist prior to running the fusion model.

The paper indicated they used a Intel CPU Xeon CPU E5-2650L v3 \@ 1.8 GHz.

2. In the ROOT/plots/images, the following files willl appear:

    - **accuracy_concat_intermediate_fusion_mitbih.png**
    - **accuracy_concat_intermediate_fusion_ptbdb.png**
    - **f1_concat_intermediate_fusion_mitbih.png**
    - **f1_concat_intermediate_fusion_ptbdb.png**
    - **loss_concat_intermediate_fusion_ptbdb.png**
    - **loss_concat_intermediate_fusion_mitbih.png**
    - **mcc_concat_intermediate_fusion_mitbih.png**
    - **mcc_concat_intermediate_fusion_ptbdb.png**

    - **accuracy_sum_intermediate_fusion_mitbih.png**
    - **accuracy_sum_intermediate_fusion_ptbdb.png**
    - **f1_sum_intermediate_fusion_mitbih.png**
    - **f1_sum_intermediate_fusion_ptbdb.png**
    - **loss_sum_intermediate_fusion_ptbdb.png**
    - **loss_sum_intermediate_fusion_mitbih.png**
    - **mcc_sum_intermediate_fusion_mitbih.png**
    - **mcc_sum_intermediate_fusion_ptbdb.png**

    - **accuracy_attention_intermediate_fusion_mitbih.png**
    - **accuracy_attention_intermediate_fusion_ptbdb.png**
    - **f1_attention_intermediate_fusion_mitbih.png**
    - **f1_attention_intermediate_fusion_ptbdb.png**
    - **loss_attention_intermediate_fusion_ptbdb.png**
    - **loss_attention_intermediate_fusion_mitbih.png**
    - **mcc_attention_intermediate_fusion_mitbih.png**
    - **mcc_attention_intermediate_fusion_ptbdb.png**

    These are plots for the various metrics.

3. After a model is trained via the test script above, the model is saved under ROOT/models/saved_models in following directories:

    - **intermediate_fusion_mitbih**
    - **intermediate_fusion_ptbdb**

    Note: It is not supported to save each of the different methods.
    
4. To test the model using the test dataset, run the following command:

    - **python3 test_intermediate_fusion.py --model concat_intermediate_fusion_ptbdb --test**
    - **python3 test_intermediate_fusion.py --model sum_intermediate_fusion_ptbdb --test**
    - **python3 test_intermediate_fusion.py --model attention_intermediate_fusion_ptbdb --test**
    
    - **python3 test_intermediate_fusion.py --model concat_intermediate_fusion_mitbih --test**
    - **python3 test_intermediate_fusion.py --model sum_intermediate_fusion_mitbih --test**
    - **python3 test_intermediate_fusion.py --model attention_intermediate_fusion_mitbih --test**