import json
import pandas as pd
import re
import string
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score


def get_accuracy_and_f1(path):
    df = pd.read_json(path, lines=True)

    label_mapping = {"support": 1, "unsupport": 0}

    y_true = df["label"].map(label_mapping).tolist()

    y_pred = []

    for pred in df["pred"]:
        matches = re.findall(r"support|unsupport", pred.strip(), re.IGNORECASE)

        if len(matches) > 0:
            pred_label = label_mapping[matches[0].lower()]
        else:
            pred_label = 0

        y_pred.append(pred_label)
        
    y_true = [int(label) for label in y_true]
    y_pred = [int(p) for p in y_pred]

    mACC = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return mACC, f1, auc_pr, recall, cm