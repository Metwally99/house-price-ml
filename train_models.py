"""Train and evaluate five PyTorch regressors on UCI Real Estate Valuation."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("DATASET_PATH", ROOT / "data" / "houses.csv"))
MODEL_PATH = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
FEATURES = ("date", "age", "distance", "stores", "latitude", "longitude")
UNIT = "10,000 TWD per ping"
SEED = 42
ARCHITECTURES = {
    "A": ("Linear regression", (), 0.01),
    "B": ("Shallow network", (16,), 0.01),
    "C": ("Two hidden layers", (32, 16), 0.01),
    "D": ("Wide hidden layer", (64,), 0.01),
    "E": ("Three hidden layers", (64, 32, 16), 0.005),
}


def make_model(hidden_dims):
    layers = []
    input_dim = len(FEATURES)
    for output_dim in hidden_dims:
        layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
        input_dim = output_dim
    return nn.Sequential(*layers, nn.Linear(input_dim, 1))


def train(model, features, target, learning_rate, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model, features):
    return model(features).flatten().numpy()


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frame = pd.read_csv(DATA_PATH)
    if not set((*FEATURES, "price")).issubset(frame.columns) and len(frame.columns) == 8:
        frame.columns = ["serial", *FEATURES, "price"]
    features = frame[list(FEATURES)].to_numpy(dtype=float)
    target = frame["price"].to_numpy(dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Training data must contain finite numeric values.")

    # Freeze the test split before training or picking a model.
    dev_x, test_x, dev_y, test_y = train_test_split(
        features, target, test_size=0.20, random_state=SEED
    )
    train_x, val_x, train_y, val_y = train_test_split(
        dev_x, dev_y, test_size=0.20, random_state=SEED
    )
    scaler = StandardScaler().fit(train_x)
    train_tensor = torch.tensor(scaler.transform(train_x), dtype=torch.float32)
    val_tensor = torch.tensor(scaler.transform(val_x), dtype=torch.float32)
    test_tensor = torch.tensor(scaler.transform(test_x), dtype=torch.float32)
    target_tensor = torch.tensor(train_y[:, None], dtype=torch.float32)
    target_mean = float(train_y.mean())
    target_scale = float(train_y.std()) or 1.0
    target_tensor = (target_tensor - target_mean) / target_scale
    baseline = DummyRegressor(strategy="mean").fit(train_x, train_y)

    report = {
        "dataset_rows": len(frame),
        "split": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "seed": SEED,
        "target_unit": UNIT,
        "selection_metric": "validation MAE",
        "models": {
            "mean": {"name": "Mean predictor", "validation": metrics(val_y, baseline.predict(val_x))}
        },
    }
    trained = {}
    for code, (name, hidden_dims, learning_rate) in ARCHITECTURES.items():
        torch.manual_seed(SEED)
        model = train(make_model(hidden_dims), train_tensor, target_tensor, learning_rate)
        trained[code] = model
        report["models"][code] = {
            "name": name,
            "validation": metrics(val_y, predict(model, val_tensor) * target_scale + target_mean),
        }

    selected = min(trained, key=lambda code: report["models"][code]["validation"]["mae"])
    report["selected_model"] = selected
    report["models"]["mean"]["test"] = metrics(test_y, baseline.predict(test_x))
    report["models"][selected]["test"] = metrics(
        test_y, predict(trained[selected], test_tensor) * target_scale + target_mean
    )
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    demo_dir = ROOT / "docs"
    demo_dir.mkdir(exist_ok=True)
    export = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "selected_model": selected,
        "models": {},
    }
    for code, model in trained.items():
        torch.save(model.state_dict(), MODEL_PATH / f"model_{code.lower()}.pth")
        export["models"][code] = [
            {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
            for layer in model
            if isinstance(layer, nn.Linear)
        ]

    (demo_dir / "models.json").write_text(json.dumps(export, separators=(",", ":")) + "\n", encoding="utf-8")
    (demo_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
