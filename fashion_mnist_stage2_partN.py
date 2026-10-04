"""
Fashion-MNIST Stage 2, Part N
Covers: CNN training, LR vs CNN comparison, confusion matrix and per-class
precision/recall, and the final one-time test set evaluation.

Run order:
  1. Set BEST_C using Logistic Regression validation performance.
  2. Run with EVAL_ON_TEST = False (default). Compare models on validation only.
  3. Once the final model is chosen, set EVAL_ON_TEST = True and run again.
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
SEED = 42
BEST_C = 0.1              # selected using validation performance from the specified C values
# sklearn C is the inverse of regularization strength: larger C means weaker regularization.
RUN_LR_COMPARISON = True  # refits LR with BEST_C to get matching metrics
FINAL_MODEL = "CNN"       # selected using validation performance
EVAL_ON_TEST = True       # final one-time test-set evaluation
EPOCHS = 20
BATCH_SIZE = 128
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

os.makedirs("report_figures", exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------------------------- data (same split as LR)
# Keep the official test data bundled and untouched until final evaluation.
(X_full, y_full), untouched_test_data = fashion_mnist.load_data()
X_full = X_full.astype("float32") / 255.0

# Identical to the baseline: 80/20, shuffled, stratified, random_state=42
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.20, random_state=SEED,
    shuffle=True, stratify=y_full)
print("Train / Val:", X_tr.shape[0], X_val.shape[0])
print("Official test set held out untouched: 10000")

# CNN input needs a channel dimension
X_tr_c = X_tr[..., np.newaxis]
X_val_c = X_val[..., np.newaxis]

# --------------------------------------------------------------------- CNN
cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
cnn.summary()

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                     restore_best_weights=True)
history = cnn.fit(X_tr_c, y_tr, validation_data=(X_val_c, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[early_stop], verbose=2)

# Learning curves (useful for the overfitting discussion)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("CNN loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("CNN accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig("report_figures/cnn_learning_curves.png", dpi=300)
plt.close()

# Final train/val metrics measured the same way (dropout off, best weights)
cnn_tr_loss, cnn_tr_acc = cnn.evaluate(X_tr_c, y_tr, verbose=0)
cnn_val_loss, cnn_val_acc = cnn.evaluate(X_val_c, y_val, verbose=0)
cnn_val_pred = np.argmax(cnn.predict(X_val_c, verbose=0), axis=1)

# ------------------------------------------------- LR for matching numbers
if RUN_LR_COMPARISON:
    Xtr_f, Xval_f = X_tr.reshape(len(X_tr), -1), X_val.reshape(len(X_val), -1)
    # Multiclass linear Logistic Regression with L2 regularization.
    lr = LogisticRegression(C=BEST_C, penalty="l2", solver="lbfgs",
                            max_iter=1000, random_state=SEED)
    lr.fit(Xtr_f, y_tr)
    # Logistic loss: negative log-probability of the true class. Do not use MSE.
    lr_tr_loss = log_loss(y_tr, lr.predict_proba(Xtr_f))
    lr_val_loss = log_loss(y_val, lr.predict_proba(Xval_f))
    lr_tr_acc = accuracy_score(y_tr, lr.predict(Xtr_f))
    lr_val_pred = lr.predict(Xval_f)
    lr_val_acc = accuracy_score(y_val, lr_val_pred)

# ------------------------------------------------------ comparison table
print("\n=== Training vs validation (for Table in Section 4) ===")
print(f"{'Model':<28}{'Train loss':>11}{'Train acc':>11}"
      f"{'Val loss':>10}{'Val acc':>9}")
if RUN_LR_COMPARISON:
    print(f"{'Logistic Regression C=' + str(BEST_C):<28}{lr_tr_loss:>11.4f}"
          f"{lr_tr_acc:>11.4f}{lr_val_loss:>10.4f}{lr_val_acc:>9.4f}")
print(f"{'CNN':<28}{cnn_tr_loss:>11.4f}{cnn_tr_acc:>11.4f}"
      f"{cnn_val_loss:>10.4f}{cnn_val_acc:>9.4f}")


# ------------------------------------- confusion matrix and per class stats
def diagnose(name, split_name, y_true, y_pred, fname):
    print(f"\n=== {name}: per class precision / recall / F1 ({split_name}) ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                digits=3))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False, values_format=".2f")
    ax.set_title(f"{name}: normalised confusion matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    # Top confused ordered pairs (true -> predicted), raw counts
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    print(f"Top 5 confusions for {name} (true -> predicted : count)")
    for i, j in idx[:5]:
        print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]} : {off[i, j]}")


diagnose("CNN", "validation", y_val, cnn_val_pred,
         "report_figures/cm_cnn_val.png")
if RUN_LR_COMPARISON:
    diagnose("Logistic Regression", "validation", y_val, lr_val_pred,
             "report_figures/cm_lr_val.png")

# ------------------------------------------- one time test set evaluation
if EVAL_ON_TEST:
    if FINAL_MODEL not in ("CNN", "Logistic Regression"):
        raise ValueError(
            'Set FINAL_MODEL to "CNN" or "Logistic Regression" before test evaluation.'
        )

    # Access and normalize the official test set only after final model selection.
    X_test, y_test = untouched_test_data
    X_test = X_test.astype("float32") / 255.0

    if FINAL_MODEL == "CNN":
        X_test_c = X_test[..., np.newaxis]
        test_loss, test_acc = cnn.evaluate(X_test_c, y_test, verbose=0)
        test_pred = np.argmax(cnn.predict(X_test_c, verbose=0), axis=1)
        loss_name = "sparse categorical cross entropy"
        figure_name = "report_figures/cm_cnn_test.png"
    elif FINAL_MODEL == "Logistic Regression":
        if not RUN_LR_COMPARISON:
            raise ValueError("RUN_LR_COMPARISON must be True to evaluate Logistic Regression.")
        Xtest_f = X_test.reshape(len(X_test), -1)
        test_pred = lr.predict(Xtest_f)
        test_loss = log_loss(y_test, lr.predict_proba(Xtest_f))
        test_acc = accuracy_score(y_test, test_pred)
        loss_name = "log loss"
        figure_name = "report_figures/cm_lr_test.png"
    print(f"\n=== FINAL TEST RESULTS ({FINAL_MODEL}, evaluated once) ===")
    print(f"Test loss ({loss_name}): {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}  (0/1 test error {1 - test_acc:.4f})")
    diagnose(FINAL_MODEL, "test", y_test, test_pred, figure_name)
else:
    print("\nTest set not touched. Set FINAL_MODEL, then set EVAL_ON_TEST = True after model selection.")
