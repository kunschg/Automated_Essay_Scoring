import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm import tqdm

from models import LSTM, ConvNet1D, SwinTransformerModelTL, TransformerModel

warnings.filterwarnings("ignore")

############
# Parameters
############


def arg_parser():
    parser = argparse.ArgumentParser()

    # General training arguments
    parser.add_argument("--local-machine", default=True, action="store_true")
    parser.add_argument("--embedder", type=str, default="w2v")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--l-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="ConvNet1D")
    parser.add_argument("--max_essay_length", type=int, default=1266)
    parser.add_argument("--input-channels", type=int, default=50)
    parser.add_argument("--do-not-save", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    # Arguments specific to convolutional models
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--skip-connections", action="store_true")
    parser.add_argument("--batch-norm", action="store_true")

    # Arguments specific to transformer models
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Arguments specific to transfer learning models
    parser.add_argument("--kernel_size_tl", type=int, default=16)
    parser.add_argument("--stride_tl", type=int, default=16)

    # Arguments specific to transfer learning models
    parser.add_argument("--hidden_size", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--batch_first", type=bool, default=True)
    args = parser.parse_args()
    return args


###################
# Training function
###################


def train(args):
    # Set seed
    torch.manual_seed(args.seed)

    # Import training and validation data sets
    print("Loading dataset...\n")

    if args.local_machine:
        train_dataset = torch.load(f"data/train_{args.embedder}.pt")
        val_dataset = torch.load(f"data/val_{args.embedder}.pt")
    else:
        train_dataset = torch.load(f"/content/train_{args.embedder}.pt")
        val_dataset = torch.load(f"/content/val_{args.embedder}.pt")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Device set-up
    if torch.cuda.is_available():
        device_var = "cuda:0"
    elif torch.backends.mps.is_available() and args.model != "LSTM":
        device_var = torch.device("mps")
    else:
        device_var = "cpu"
    device = torch.device(device_var)
    print(f"The current processor is: {device}\n")

    # Model initialization
    if args.model.startswith("ConvNet1D"):
        model = eval(args.model)(
            args.input_channels,
            args.depth,
            args.kernel_size,
            args.max_essay_length,
            args.skip_connections,
            args.batch_norm,
        )
        model_name = f"{args.model}_depth{args.depth}_seed{args.seed}"

    elif args.model == "TransformerModel":
        model = eval(args.model)(
            args.input_channels,
            args.max_essay_length,
            args.num_heads,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
        )
        model_name = f"{args.model}_numlayers{args.num_layers}_seed{args.seed}"

    elif args.model.startswith("SwinTransformerModelTL"):
        pretrained_model = models.swin_t(pretrained=True)

        for param in pretrained_model.parameters():
            param.requires_grad = False

        model = eval(args.model)(
            args.input_channels,
            args.max_essay_length,
            args.kernel_size_tl,
            args.stride_tl,
            pretrained_model,
        )
        model_name = f"{args.model}_seed{args.seed}"

    elif args.model == "LSTM":
        model = eval(args.model)(
            args.batch_size,
            args.input_channels,
            args.max_essay_length,
            args.hidden_size,
            args.num_layers,
            args.bidirectional,
            args.batch_first,
        )
        model_name = f"{args.model}_numlayers{args.num_layers}_seed{args.seed}"

    else:
        raise Exception("Invalid model")

    model = model.to(device)

    # Optimizer set-up
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.l_rate, momentum=args.momentum
        )
    else:
        raise Exception("Invalid optimizer")

    # Initialize training metrics
    training_losses, val_losses, val_maes = [], [], []
    training_loss, val_loss, val_mae = 0, 0, 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        print(f"Epoch [{0}/{args.n_epochs}]")

        print("Training...")
        for essays, essay_sets, grades in tqdm(train_loader):
            essays, essay_sets, grades = (
                essays.to(device),
                essay_sets.to(device),
                grades.to(device),
            )
            grades_pred = model(essays, essay_sets)
            loss = criterion(grades_pred, grades)
            training_loss += loss.item()
        training_loss = training_loss / len(train_loader)
        training_losses.append(training_loss)

        print("Validation...")
        for essays, essay_sets, grades in tqdm(val_loader):
            essays, essay_sets, grades = (
                essays.to(device),
                essay_sets.to(device),
                grades.to(device),
            )
            grades_pred = model(essays, essay_sets)
            loss = criterion(grades_pred, grades)
            val_loss += loss.item()
            val_mae += torch.mean(torch.abs(grades_pred - grades)).item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_mae = val_mae / len(val_loader)
        val_maes.append(val_mae)

    print(
        f"Training loss: {training_loss:.4f}, Validation loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}\n"
    )

    # Initialize early stopping metrics
    best_val_mae = val_mae
    wait = 0

    # Training process
    for epoch in range(1, args.n_epochs + 1):
        print(f"Epoch [{epoch}/{args.n_epochs}]")
        training_loss, val_loss, val_mae = 0, 0, 0

        # Training loop
        print("Training...")
        for essays, essay_sets, grades in tqdm(train_loader):
            essays, essay_sets, grades = (
                essays.to(device),
                essay_sets.to(device),
                grades.to(device),
            )
            grades_pred = model(essays, essay_sets)
            loss = criterion(grades_pred, grades)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        training_loss = training_loss / len(train_loader)
        training_losses.append(training_loss)

        # Validation loop
        print("Validation...")
        with torch.no_grad():
            for essays, essay_sets, grades in tqdm(val_loader):
                essays, essay_sets, grades = (
                    essays.to(device),
                    essay_sets.to(device),
                    grades.to(device),
                )
                grades_pred = model(essays, essay_sets)
                loss = criterion(grades_pred, grades)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(grades_pred - grades)).item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_mae = val_mae / len(val_loader)
        val_maes.append(val_mae)

        print(
            f"Training loss: {training_loss:.4f}, Validation loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}\n"
        )

        # Check for early stopping
        if val_mae < best_val_mae - args.min_delta:
            best_val_mae = val_mae
            wait = 0
            if not args.do_not_save:
                torch.save(model.state_dict(), f"models/{model_name}.pth")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    print(f"Best validation MAE: {best_val_mae:.4f}")

    if not args.do_not_save:
        metrics = np.concatenate([[training_losses], [val_losses], [val_maes]], axis=0)
        np.save("outputs/" + model_name, metrics)


#############
# File runner
#############

args = arg_parser()
train(args)
