import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def PlotCurves(train_accuracies, train_losses, valid_accuracies, 
    valid_losses, file_name, test_accuracies=None, test_losses=None):

    num_epochs = len(train_losses)
    xvalues = np.arange(0, num_epochs, 1).astype('str')
    fig, axs = plt.subplots(figsize=(8,8))
    train_loss = axs.plot(xvalues, train_losses, label="Training Loss")
    valid_loss = axs.plot(xvalues, valid_losses, label="Validation Loss")
    axs.legend(loc="upper right")
    axs.set(xlabel='epoch', ylabel='Loss', title='Loss Curve')
    if num_epochs > 10:
        axs.set_xticks(range(0, num_epochs, 5))
        axs.set_xticklabels([i for i in range(0, num_epochs, 5)])

    plots_dir = (Path(__file__).parent).joinpath('images')
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
    plots_file_name = plots_dir.joinpath("loss_" + file_name + ".png")
    fig.savefig(plots_file_name)

    fig, axs = plt.subplots(figsize=(10,10))
    train_accuracy = axs.plot(xvalues, train_accuracies, label="Training Accuracy")
    valid_accuracy = axs.plot(xvalues, valid_accuracies, label="Validation Accuracy")
    axs.legend(loc="upper left")
    axs.set(xlabel='epoch', ylabel='Accuracy', title='Accuracy Curve')
    if num_epochs > 10:
        axs.set_xticks(range(0, num_epochs, 5))
        axs.set_xticklabels([i for i in range(0, num_epochs, 5)])
    
    plots_file_name = plots_dir.joinpath("accuracy_" + file_name + ".png")
    fig.savefig(plots_file_name)

def PlotClassDistribution(class_label_map, class_info, file_name):
    fig, axs = plt.subplots(figsize=(12,12))

    class_names = [ k for k, v in sorted(class_label_map.items(), key=lambda i: i[1])]
    class_counts = [v for k, v in sorted(class_info.items(), key=lambda i: i[0])]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(class_names)))
    axs.bar(class_names, class_counts, color=colors, label=class_names)
    axs.set(ylabel='Counts for each class', xlabel='Class Names', title="Class Distribution")
    for l in axs.get_xticklabels():
        l.set_rotation(45)
    axs.legend(loc="upper right")

    plots_dir = (Path(__file__).parent).joinpath('images')
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
    plots_file_name = plots_dir.joinpath("distribution_" + file_name + ".png")
    fig.savefig(plots_file_name)

def PlotMCCCurves(train_mcc_scores, valid_mcc_scores, file_name, test_scores=None):

    num_epochs = len(train_mcc_scores)
    xvalues = np.arange(0, num_epochs, 1).astype('str')
    fig, axs = plt.subplots(figsize=(8,8))
    train_mcc = axs.plot(xvalues, train_mcc_scores, label="Training MCC")
    valid_mcc = axs.plot(xvalues, valid_mcc_scores, label="Validation MCC")
    axs.legend(loc="upper right")
    axs.set(xlabel='epoch', ylabel='MCC', title='Matthew Correlation Coefficient')
    if num_epochs > 10:
        axs.set_xticks(range(0, num_epochs, 5))
        axs.set_xticklabels([i for i in range(0, num_epochs, 5)])

    plots_dir = (Path(__file__).parent).joinpath('images')
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
    plots_file_name = plots_dir.joinpath("mcc_" + file_name + ".png")
    fig.savefig(plots_file_name)

def PlotF1Curves(train_f1_scores, valid_f1_scores, file_name, test_scores=None):

    num_epochs = len(train_f1_scores)
    xvalues = np.arange(0, num_epochs, 1).astype('str')
    fig, axs = plt.subplots(figsize=(8,8))
    train_mcc = axs.plot(xvalues, train_f1_scores, label="Training F1")
    valid_mcc = axs.plot(xvalues, valid_f1_scores, label="Validation F1")
    axs.legend(loc="upper right")
    axs.set(xlabel='epoch', ylabel='F1', title='F1 Score')
    if num_epochs > 10:
        axs.set_xticks(range(0, num_epochs, 5))
        axs.set_xticklabels([i for i in range(0, num_epochs, 5)])

    plots_dir = (Path(__file__).parent).joinpath('images')
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
    plots_file_name = plots_dir.joinpath("f1_" + file_name + ".png")
    fig.savefig(plots_file_name)
    
