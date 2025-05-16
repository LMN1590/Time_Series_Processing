import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time
from preprocess.const import ROBUST_SCALE_CONST

from model_module import PatientModelModule
model = PatientModelModule.load_from_checkpoint(
    checkpoint_path="inference_utils/Switch_Former_epoch(29)_step(0960)_val_0.2882.ckpt",
    hparams_file=f"inference_utils/hparams.yaml"
).eval()

LIMIT = 730

df = pd.read_csv("data/unnorm_data_reduced_shift.csv")
df = df[df.groupby('PatientId')['PatientId'].transform('size') >= 4]

pt_ids = df["PatientId"].unique()
# random.shuffle(pt_ids)
df = df.set_index("PatientId").loc[pt_ids].reset_index()
df["mean"] = np.nan
df["std"] = np.nan

start_time = time.time()
count_instances = 0

for pt in tqdm(pt_ids):
    df_pt = df[df["PatientId"]==pt]
    row_length = df_pt.shape[0]
    row_index = list(df_pt.index)[-(row_length-4+1):]

    data = []
    
    for idx in range(3,row_length):
        serie = list(range(idx-4+1,idx+1))
        full_data = df_pt.iloc[serie].reset_index()
        full_data = full_data.drop(full_data.columns[[0,1,2,3,4]], axis=1).drop(["mean","std","Target","Target_shift_1"],axis=1).values.astype(float)
        input_data = torch.from_numpy(full_data)
        data.append(input_data)
        count_instances+=1
        
    input_data = torch.stack(data).float()
    with torch.no_grad():
        res = model(input_data)
        
    mean = res[0]
    var = res[1]
    # # region Standard Scaler
    # center = STANDARD_SCALE_CONST["mean"]
    # scale = STANDARD_SCALE_CONST["scale"]
    # # endregion

    # region Robust Scaler
    center = ROBUST_SCALE_CONST["center"][-1]
    scale = ROBUST_SCALE_CONST["scale"][-1]
    # endregion
    mean_unnorm = mean*scale+center
    var_unnorm = var*(scale**2)

    mean_unlog = torch.exp(mean_unnorm + (var_unnorm)/2)-1
    var_unlog = torch.exp(2*mean_unnorm)*(torch.exp(2*var_unnorm) - torch.exp(var_unnorm))
    std_unlog = torch.sqrt(var_unlog)
    
    df.loc[row_index,"mean"] = mean_unlog.numpy()
    df.loc[row_index,"std"] = std_unlog.numpy()
    
print((time.time()-start_time)/count_instances)
df["Lost?"] = df["Target_shift_1"]>LIMIT
    
df.to_csv("./data/unnorm_data_mean_std_dummy.csv")