import os, time, numpy as np, pandas as pd
from comet_ml import Optimizer, Experiment
from comet_ml.integration.pytorch import watch
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve

# ─── CONFIG
WORKSPACE, PROJECT_NAME = "vanwpham", "mnist-comet-demo"
SWEEP_CONFIG = {
    "algorithm": "bayes",
    "parameters": {
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-1},
        "momentum":      {"type": "float", "min": 0.5,  "max": 0.99},
        "batch_size":    {"type": "categorical", "values": ["32", "64", "128"]},
    },
    "spec": {"metric": {"name": "val_accuracy", "goal": "maximize"},
             "trials": 1,          # ≤ 8 completed; no bù nếu lỗi
             "retryLimit": 0
            }
}

T = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])
TRAIN_DS = datasets.MNIST("data", train=True,  download=True, transform=T)
TEST_DS  = datasets.MNIST("data", train=False, download=True, transform=T)

TRAIN_COUNTS = pd.Series(TRAIN_DS.targets.numpy()).value_counts().sort_index()
TEST_COUNTS  = pd.Series(TEST_DS.targets.numpy()).value_counts().sort_index()
DATA_SUMMARY = pd.DataFrame({"label": range(10),
                             "train_count": TRAIN_COUNTS.values,
                             "test_count":  TEST_COUNTS.values})

# GRID_IMG = make_grid(torch.stack([TRAIN_DS[i][0] for i in range(16)]),
#                      nrow=4).permute(1, 2, 0).numpy()

# ─── MODEL
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1   = nn.Linear(32*5*5, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.flatten(1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# ─── TRAIN LOOP
def train_fn(experiment: Experiment, params: dict):
    bs = int(params["batch_size"]); lr = params["learning_rate"]; mom = params["momentum"]
    experiment.set_name(f"lr={lr:.1e}_mom={mom:.2f}_bs={bs}")
    experiment.add_tag("sweep-demo")
    experiment.log_parameters(params)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, opt = SimpleCNN().to(dev), optim.SGD(SimpleCNN().parameters(), lr=lr, momentum=mom)
    crit = nn.CrossEntropyLoss(); watch(model)

    TL = DataLoader(TRAIN_DS, batch_size=bs, shuffle=True)
    VL = DataLoader(TEST_DS,  batch_size=bs, shuffle=False)

    # dataset logs
    experiment.log_table(filename="dataset_summary.csv", tabular_data=DATA_SUMMARY)
    experiment.log_histogram_3d(TRAIN_COUNTS.values, name="train_dist", step=0)
    experiment.log_histogram_3d(TEST_COUNTS.values,  name="test_dist",  step=0)
    
    # experiment.log_image(GRID_IMG, name="sample_images")

    tr_loss, va_loss, va_acc = [], [], []
    best_acc, best_ckpt = 0, None
    for epoch in range(1, 6):
        # train
        model.train(); s, c = 0.0, 0
        for X, y in TL:
            X, y = X.to(dev), y.to(dev); opt.zero_grad()
            out = model(X); loss = crit(out, y); loss.backward(); opt.step()
            s += loss.item(); c += (out.argmax(1) == y).sum().item()
        tr_loss.append(s / len(TL)); train_acc = c / len(TL.dataset)

        # val
        model.eval(); s, c, pr, tr, p0 = 0.0, 0, [], [], []
        with torch.no_grad():
            for X, y in VL:
                X, y = X.to(dev), y.to(dev); out = model(X)
                s += crit(out, y).item(); p = out.argmax(1)
                c += (p == y).sum().item()
                pr.extend(p.cpu()); tr.extend(y.cpu())
                p0.extend(torch.softmax(out,1)[:,0].cpu())
        va_loss.append(s / len(VL)); v_acc = c / len(VL.dataset); va_acc.append(v_acc)

        # logs
        experiment.log_confusion_matrix(y_true=tr, y_predicted=pr, step=epoch)
        fpr, tpr, _ = roc_curve(np.array(tr)==0, np.array(p0))
        experiment.log_curve(name="ROC_class0", x=fpr, y=tpr, step=epoch)
        experiment.log_metrics({"train_loss": tr_loss[-1], "train_acc": train_acc,
                                "val_loss":   va_loss[-1], "val_acc": v_acc}, step=epoch)

        if v_acc > best_acc:
            best_acc, best_ckpt = v_acc, f"best_{experiment.get_key()}_e{epoch}.pth"
            torch.save(model.state_dict(), best_ckpt)
            experiment.log_model("best_checkpoint", best_ckpt)

    # — new: log curves instead of log_chart —
    epochs = list(range(1, 6))
    experiment.log_curve(name="train_loss_curve", x=epochs, y=tr_loss, step=5)
    experiment.log_curve(name="val_loss_curve",   x=epochs, y=va_loss, step=5)
    experiment.log_curve(name="val_acc_curve",    x=epochs, y=va_acc, step=5)

    # table HP + best score
    hp_df = pd.DataFrame([{"batch_size": bs, "learning_rate": lr,
                           "momentum": mom, "best_val_acc": best_acc}])
    experiment.log_table(filename="hp_metrics.csv", tabular_data=hp_df)
    experiment.log_metric("best_val_acc", best_acc, step=5)

    experiment.log_text(text=f"# MNIST CometML Demo\n\nBest Val Acc: {best_acc:.4%}")
    if best_ckpt: experiment.log_model("final_model", best_ckpt)
    experiment.end()

def safe_train(experiment, params):
    try:
        train_fn(experiment, params)          # hàm huấn luyện gốc
    except Exception as e:
        experiment.log_other("exception", str(e))   # ghi lỗi để tra cứu
        raise                                       # để hiện full trace
    finally:
        if experiment.alive:                       # đảm bảo kết thúc
            experiment.end()

# ─── SWEEP RUNNER

if __name__ == "__main__":
    # Thay vì sweep
    exp = Experiment(api_key=os.getenv("COMET_API_KEY"),
                    workspace=WORKSPACE,
                    project_name=PROJECT_NAME)
    train_fn(exp, {"learning_rate": 1e-3, "momentum": 0.9, "batch_size": "64"})


# if __name__ == "__main__":
#     optzr = Optimizer(api_key=os.getenv("COMET_API_KEY"), config=SWEEP_CONFIG)
#     for exp in optzr.get_experiments(workspace=WORKSPACE, project_name=PROJECT_NAME):
#         safe_train(exp, exp.params)
