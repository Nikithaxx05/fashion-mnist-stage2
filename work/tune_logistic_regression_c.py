import gzip
import struct
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


SEED = 42
C_VALUES = [0.01, 0.1, 1.0, 10.0]
DATA_DIR = Path("/Users/satyapulapa/.keras/datasets/fashion-mnist")


def load_training_data():
    """Load only the official Fashion-MNIST training files, never the test files."""
    with gzip.open(DATA_DIR / "train-images-idx3-ubyte.gz", "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError("Unexpected Fashion-MNIST training image file format.")
        images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(
            count, rows, cols
        )

    with gzip.open(DATA_DIR / "train-labels-idx1-ubyte.gz", "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError("Unexpected Fashion-MNIST training label file format.")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)

    if len(images) != count or len(labels) != count:
        raise ValueError("Fashion-MNIST training image and label counts do not match.")
    return images.astype("float32") / 255.0, labels


X_full, y_full = load_training_data()
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full,
    y_full,
    test_size=0.20,
    random_state=SEED,
    shuffle=True,
    stratify=y_full,
)
Xtr_f = X_tr.reshape(len(X_tr), -1)
Xval_f = X_val.reshape(len(X_val), -1)

print(f"Train / validation: {len(X_tr)} / {len(X_val)}", flush=True)
print("Test files opened: no", flush=True)
print("C,train_log_loss,train_accuracy,val_log_loss,val_accuracy,converged", flush=True)

for c_value in C_VALUES:
    model = LogisticRegression(
        C=c_value,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(Xtr_f, y_tr)

    convergence_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    converged = not convergence_warnings

    train_prob = model.predict_proba(Xtr_f)
    val_prob = model.predict_proba(Xval_f)
    train_loss = log_loss(y_tr, train_prob)
    val_loss = log_loss(y_val, val_prob)
    train_acc = accuracy_score(y_tr, model.predict(Xtr_f))
    val_acc = accuracy_score(y_val, model.predict(Xval_f))

    print(
        f"{c_value},{train_loss:.8f},{train_acc:.8f},"
        f"{val_loss:.8f},{val_acc:.8f},{converged}",
        flush=True,
    )
    if convergence_warnings:
        for warning in convergence_warnings:
            print(f"ConvergenceWarning for C={c_value}: {warning.message}", flush=True)
