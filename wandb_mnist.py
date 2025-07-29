import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision                                  
from torchvision import datasets, transforms
from torchvision.utils import make_grid             
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

import wandb

# 1. SWEEP CONFIGURATION (Hyperparameter optimization)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},
        'momentum':      {'distribution': 'uniform',             'min': 0.5,  'max': 0.99},
        'batch_size':    {'values': [32, 64, 128]},
        'epochs':        {'value': 5}
    }
}

# 2. TRAINING FUNCTION (for each sweep trial)
def train():
    # Init run (project inherited from sweep)
    wandb.init(tags=["demo", "full-featured"], notes="Full-featured MNIST demo")
    config = wandb.config

    # DATA LOADING & EXPLORATION
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)

    # Dataset summary table + charts
    train_counts = pd.Series(train_ds.targets.numpy()).value_counts().sort_index()
    test_counts  = pd.Series(test_ds.targets.numpy()).value_counts().sort_index()
    df_summary = pd.DataFrame({
        "label":       list(range(10)),
        "train_count": train_counts.values,
        "test_count":  test_counts.values
    })
    table = wandb.Table(dataframe=df_summary)
    wandb.log({
        "dataset_table": wandb.Table(dataframe=df_summary),
        "bar_train":     wandb.plot.bar(table, "label", "train_count", title="Train Distribution"),
        "bar_test":      wandb.plot.bar(table, "label", "test_count",  title="Test Distribution"),
        "hist_pixels":   wandb.Histogram(np.array(train_ds.data).flatten()),
        "image_grid":    wandb.Image(make_grid(torch.stack([train_ds[i][0] for i in range(16)]),
                                              nrow=4),
                                     caption="Sample Images")
    })

    # MODEL & OPTIMIZER
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,16,3,1)
            self.conv2 = nn.Conv2d(16,32,3,1)
            self.fc1   = nn.Linear(32*5*5,128)
            self.fc2   = nn.Linear(128,10)
        def forward(self,x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x,2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x,2)
            x = x.view(-1,32*5*5)
            x = nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    model     = SimpleCNN()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.learning_rate,
                          momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, log="all", log_freq=50)

    # DATALOADERS
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False)

    best_val_acc = 0.0
    for epoch in range(1, config.epochs+1):
        # TRAIN
        model.train()
        t_loss, t_correct = 0.0, 0
        start = time.time()
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_correct += (out.argmax(1)==y).sum().item()
        train_time = time.time() - start
        train_acc  = t_correct / len(train_loader.dataset)

        # VALIDATE
        model.eval()
        v_loss, v_correct = 0.0, 0
        preds, trues = [], []
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                v_loss += criterion(out, y).item()
                p = out.argmax(1)
                v_correct += (p==y).sum().item()
                preds.extend(p.cpu().numpy())
                trues.extend(y.cpu().numpy())
        val_acc   = v_correct / len(test_loader.dataset)
        precision = precision_score(trues, preds, average='macro')
        recall    = recall_score(trues, preds, average='macro')
        f1        = f1_score(trues, preds, average='macro')

        # Charts
        cm = wandb.plot.confusion_matrix(
            probs=None, y_true=trues, preds=preds, class_names=[str(i) for i in range(10)]
        )
        fpr, tpr, _ = roc_curve([t==0 for t in trues], [p==0 for p in preds])
        roc_line = wandb.plot.line_series(
            xs=fpr, ys=[tpr], keys=["class_0_tpr"], title="ROC curve (class 0)"
        )

        wandb.log({
            "epoch":           epoch,
            "train_loss":      t_loss/len(train_loader),
            "train_acc":       train_acc,
            "val_loss":        v_loss/len(test_loader),
            "val_acc":         val_acc,
            "precision":       precision,
            "recall":          recall,
            "f1_score":        f1,
            "train_time":      train_time,
            "confusion_matrix": cm,
            "roc_curve":       roc_line
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = f"ckpt_e{epoch}_acc{val_acc:.4f}.pt"
            torch.save(model.state_dict(), ckpt)
            art = wandb.Artifact("best-checkpoint", type="model",
                                 metadata={"epoch": epoch, "val_acc": val_acc})
            art.add_file(ckpt)
            wandb.log_artifact(art)

    # Final model artifact
    final_art = wandb.Artifact("final-model", type="model",
                               metadata={"best_val_acc": best_val_acc})
    final_art.add_file(ckpt)
    wandb.log_artifact(final_art)

    # Report artifact
    report = wandb.Artifact("training-report", type="report")
    with open("report.md", "w") as f:
        f.write(f"# Report\n\nBest val_acc: {best_val_acc:.4f}\n\n")
    report.add_file("report.md")
    wandb.log_artifact(report)

    wandb.finish()


# 3. RUN THE SWEEP
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="mnist-full-demo")
    wandb.agent(sweep_id, function=train, count=20)
