import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time
from preprocess.const import ROBUST_SCALE_CONST

def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score given two boolean numpy arrays.
    
    Parameters:
    y_true (np.ndarray): Ground truth (boolean array).
    y_pred (np.ndarray): Predicted values (boolean array).
    
    Returns:
    dict: Precision, Recall, and F1 score.
    """
    # Ensure the inputs are boolean arrays
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    tp = np.sum(y_true & y_pred)  # Logical AND for true positives
    fp = np.sum(~y_true & y_pred)  # Logical AND for false positives
    fn = np.sum(y_true & ~y_pred)  # Logical AND for false negatives
    tn = np.sum(~y_true & ~y_pred)

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score: Harmonic mean of precision and recall
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1,"true_positive":tp,"false_positive":fp,"false_negative":fn,"true_negative":tn}


df = pd.read_csv("data/resulu/unnorm_data_filtered_lost_mean_std.csv")
df = df[df.groupby('PatientId')['PatientId'].transform('size') >= 4]

pt_ids = df["PatientId"].unique()
# random.shuffle(pt_ids)
df = df.set_index("PatientId").loc[pt_ids]

train_num = int(len(pt_ids)*0.6)
val_num = 0
test_num = len(pt_ids) - train_num - val_num


border1s = [0, train_num, train_num + val_num]
border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

[train_ids,val_ids, test_ids] = [pt_ids[borders[0]:borders[1]] for borders in zip(border1s,border2s)]

test_data = df.loc[test_ids]

res_csv = pd.DataFrame(columns=["scale","precision","recall","f1_score","true_positive","false_positive","false_negative","true_negative","average_scale"])

lost_count = {
    1:0,
    1.5:0,
    2:0,
    2.5:0,
    3:0,
    3.5:0,
    4:0,
    4.5:0,
    5:0
}


for i in tqdm(lost_count.keys()):
    for scale in range(0,50):
        max_lim = test_data["mean_pred"] + test_data["std_pred"]*scale
        pred = test_data["Target_shift_1"] > max_lim
        metric = calculate_metrics(test_data[f"Lost_lim_{i}?"],pred)
        
        new_row = {
            "scale":scale,
            **metric,
            "average_scale":(test_data["std_pred"]*scale).mean()
        }
        # print(new_row)
        res_csv.loc[scale-20] = new_row
        
    res_csv.to_csv(f"./data/scale_result_filtered_{i}.csv")