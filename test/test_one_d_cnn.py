
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import time

import numpy as np
from data.dataloaders import pre_process_data, get_dataloader
from config.config_setup import LocalVariables
from models.one_d_cnn_transformer import OneDCnnTransformer, NoamScheduler
from torch.optim.lr_scheduler import LambdaLR
from plots.plots import PlotCurves, PlotF1Curves, PlotMCCCurves
from sklearn.metrics import matthews_corrcoef, f1_score
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(f"{__file__}: --model \"1d_cnn_mitbih|1d_cnn_ptbdb\" --train|--valid|--test")
    parser.add_argument("--train", action='store_true', help='Set the flag to True')
    parser.add_argument("--valid", action='store_true', help='Set the flag to True')
    parser.add_argument("--test", action='store_true', help='Set the flag to True')
    parser.add_argument("--model", nargs='*', type=str, default=["1d_cnn_mitbih"])
    parser.add_argument("--reduce", action='store_true', help="Set the flag to True")

    parsed_args = parser.parse_args()
    return parsed_args

def test_code_segment(model, weights, test_dataloader, device):
    total = 0.0
    correct = 0.0
    all_test_labels = []
    all_test_predictions = []

    print(f"weights {weights}")
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.eval()
    model.to(device)

    for batch_idx, (inputs, labels) in enumerate(test_dataloader, start=1):
        with torch.no_grad():
            total += labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            # save the labels and predicted values
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    mcc = matthews_corrcoef(all_test_labels, all_test_predictions)
    f1 = f1_score(all_test_labels, all_test_predictions, average='weighted')

    print(f"------- Test Metrics ----------")
    print(f"Accuracy: {accuracy} MCC: {mcc} f1 score: {f1}")


def main(cnn_config, transformer_config, local_config):
    parsed_args = parse_args()

    # training is default if no parameters are specified
    is_train = True
    is_test = False
    is_valid = False
    is_reduce = False

    # perform testing only using test data
    if parsed_args.train == False and parsed_args.test == True and parsed_args.valid == False:
        is_train = False
        is_test = True
    # perform training and validation
    elif parsed_args.valid == True and parsed_args.test == True:
        is_valid = True
        is_test = True
    elif parsed_args.valid == True:
        is_valid = True

    if parsed_args.reduce == True:
        is_reduce = True

    model = []
    for m in parsed_args.model:
            model.append(m)
    print(f"Model: {model}")
    
    # add to the local_config the type of model user wants to process
    valid_models = ["1d_cnn_mitbih", "1d_cnn_ptbdb"]

    for m in model:
        assert m in valid_models, f"{m} is not a valid model supported"


    for m in model:
        # obtain the type of data using to obtain dataloader
        data_info = m.split('_')
        data_name = data_info[-1]
        train_parquet = None
        test_parquet = None
        if data_name == "mitbih" and is_reduce == True:
            train_parquet = local_config.parquet_files[data_name + "_train_reduced"]
            test_parquet = local_config.parquet_files[data_name + "_test_reduced"]
        else:
            train_parquet = local_config.parquet_files[data_name + "_train"]
            test_parquet = local_config.parquet_files[data_name + "_test"]

        npy_file_name = str(project_root.joinpath("data").joinpath(data_name).joinpath(data_name + "_class_weights.npy"))
        class_weights = torch.tensor(np.load(npy_file_name), dtype=torch.float32)

        batch_size = None
        if "1d_cnn" in m:
            batch_size = local_config.one_d_cnn_info["batch_size"]
        else:
            batch_size = local_config.hybrid_info["batch_size"]

        if 'mitbih' in data_name:
            transformer_config["num_classes"] = 5
        else:
            transformer_config["num_classes"] = 2

        if is_train == True and is_valid == True:
            train_dataloader, valid_dataloader = get_dataloader(train_parquet, batch_size, local_config.dataloader_workers, split=True)
        elif is_train == True:
            train_dataloader = get_dataloader(train_parquet, batch_size, local_config.dataloader_workers)
        if is_test:
            test_dataloader = get_dataloader(test_parquet, batch_size, local_config.dataloader_workers)

        # obtain one sample of batch from dataloader
        if is_train == True:
            train_iteration = next(iter(train_dataloader))
            print(f"Testing for file: {train_parquet}")
            print(f"train iteration shape for first batch features: {train_iteration[0].shape}")
            print(f"train iteration shape for first batch labels:  {train_iteration[1].shape}")
        if is_test == True:
            test_iteration = next(iter(test_dataloader))
            print(f"Testing for file: {test_parquet}")
            print(f"test iteration shape for first batch features: {test_iteration[0].shape}")
            print(f"test iteration shape for first batch labels: {test_iteration[1].shape}")

        if is_valid == True:
            valid_iteration = next(iter(valid_dataloader))
            print(f"valid iteration shape for first batch features: {valid_iteration[0].shape}")

        model_file_dir = Path(local_config.saved_models)
        model_file_dir.mkdir(parents=True, exist_ok=True)
        model_file_name = str(model_file_dir.joinpath("1d_cnn_transformer_"+data_name+".pth"))

        device=None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = OneDCnnTransformer(cnn_config, transformer_config, batch_size)

        if is_train == True:
            model.to(device)

            print(f"model device: {model.device}")

            print(f"-----Test one_d_cnn_transformer for: {data_name}------")
            class_weights = class_weights.to(device)
            criterion = None
            if is_reduce == True:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)

            beta_1 = local_config.one_d_cnn_optim["beta_1"]
            beta_2 = local_config.one_d_cnn_optim["beta_2"]

            optimizer = optim.Adam(model.parameters(), lr=local_config.one_d_cnn_info["learning_rate"], \
                betas=(beta_1, beta_2))

            # set the Noam Scheduler
            scheduler = NoamScheduler(optimizer, d_model=cnn_config["d_model"], warmup_steps=local_config.one_d_cnn_optim["warm_up"])

            # check if each param in device
            #for name, param in model.named_parameters():
            #       print(f"Parameter {name} is on device: {param.device}")

            # set the training losses list
            training_time = [time.time()]
            epochs = local_config.one_d_cnn_info["epochs"]
            training_losses = []
            training_accuracies = []
            valid_accuracies = []
            valid_losses = []
            mcc_train_scores = []
            mcc_valid_scores = []
            f1_train_scores = []
            f1_valid_scores = []
            best_acc = 0.0
            train_time = 0.0

            for num_epoch, epoch in enumerate(range(epochs), start=1):
                model.train()

                train_loss = 0.0
                total = 0.0
                correct = 0.0
                avg_loss = 0.0
    
                all_train_labels = []
                all_train_predictions = []

                for batch_idx, (inputs, labels) in enumerate(train_dataloader, start=1):
                    total += labels.size(0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()

                    # save the labels and predicted values
                    all_train_labels.extend(labels.cpu().numpy())
                    all_train_predictions.extend(predicted.cpu().numpy())

                avg_loss = train_loss / len(train_dataloader)
                accuracy = correct / total
                mcc = matthews_corrcoef(all_train_labels, all_train_predictions)
                f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
                
                print(f"Epoch: {num_epoch}, train accuracy: {accuracy} avg_loss: {avg_loss}")

                training_losses.append(avg_loss)
                training_accuracies.append(accuracy)
                mcc_train_scores.append(mcc)
                f1_train_scores.append(f1)

                if is_valid == True:
                    val_loss = 0.0
                    total = 0.0
                    correct = 0.0
                    avg_loss = 0.0
        
                    all_valid_labels = []
                    all_valid_predictions = []

                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, labels) in enumerate(valid_dataloader, start=1):
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs, _ = model(inputs)
                            loss = criterion(outputs, labels.long())
                            val_loss += loss.item()
                            total += labels.size(0)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == labels).sum().item()

                            # save the labels and predicted values
                            all_valid_labels.extend(labels.cpu().numpy())
                            all_valid_predictions.extend(predicted.cpu().numpy())

                        avg_loss = val_loss / len(valid_dataloader)
                        accuracy = correct / total
                        mcc = matthews_corrcoef(all_valid_labels, all_valid_predictions)
                        f1 = f1_score(all_valid_labels, all_valid_predictions, average='weighted')

                        valid_losses.append(avg_loss)
                        valid_accuracies.append(accuracy)
                        mcc_valid_scores.append(mcc)
                        f1_valid_scores.append(f1)

                    if accuracy > best_acc:
                        best_acc = accuracy
                        torch.save(model.state_dict(), model_file_name)

                    print(f"Epoch: {num_epoch}, valid accuracy: {accuracy} avg_loss: {avg_loss}")

            train_time = time.time()
            if is_valid == True:
                PlotCurves(training_accuracies, training_losses, valid_accuracies, valid_losses, m)
                PlotMCCCurves(mcc_train_scores, mcc_valid_scores, m)
                PlotF1Curves(f1_train_scores, f1_valid_scores, m)
                print(f"Training and Validation time: {train_time - training_time[-1]:.2f}")
            else:
                torch.save(model.state_dict(), model_file_name)
                print(f"Training time: {train_time - training_time[-1]:.2f}")

        if is_test == True: # this is the testing of model
            start_test_time = time.time()
            test_code_segment(model, model_file_name, test_dataloader, device)
            end_test_time = time.time()
            print(f"Testing time: {end_test_time - start_test_time:.2f}")


if __name__ == '__main__':
    
    config_file = project_root.joinpath("config").joinpath("configuration.yaml")
    print(f"config_file {config_file}")
    local_config = LocalVariables(yaml_config_file=config_file)
    cnn_config = {}
    transformer_config = {}

    transformer_config["n_head"] = local_config.n_head
    transformer_config["n_layers"] = local_config.n_layers
    transformer_config["d_hid"] = local_config.d_hid
    transformer_config["dropout"] = local_config.one_d_cnn_info["dropout"]
    transformer_config["embedding_dim"] = local_config.d_model
    transformer_config["num_classes"] = 0

    cnn_config["n_conv"] = local_config.n_conv
    cnn_config["n_filters"] = local_config.n_filters
    cnn_config["d_model"] = local_config.d_model
    cnn_config["d_proj"] = local_config.d_proj
    cnn_config["in_channels"] = 1

    main(cnn_config, transformer_config, local_config)