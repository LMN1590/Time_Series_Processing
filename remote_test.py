from modal import App,Image,gpu,Mount,Volume

model_name = "switch_transformer"
app = App(f'Testing {model_name}')

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)

@app.function(
    image=image,
    gpu = "a10g",
    timeout=86400,
    retries=0,
    mounts = [
        Mount.from_local_dir("configs",remote_path="/root/configs"),
        Mount.from_local_dir("model",remote_path="/root/model"),
        Mount.from_local_dir("train_util",remote_path="/root/train_util"),
        Mount.from_local_dir("loss",remote_path="/root/loss"),
        Mount.from_local_dir("preprocess",remote_path="/root/preprocess"),
        
        Mount.from_local_file("const.py","/root/const.py"),
        Mount.from_local_file("data_module.py","/root/data_module.py"),
        Mount.from_local_file("main.py","/root/main.py"),
        Mount.from_local_file("model_module.py","/root/model_module.py"),
        Mount.from_local_file("remote_test.py","/root/remote_test.py")
        
    ],
    volumes={
        "/root/saved": Volume.from_name("retention_log_final"),
        "/root/data": Volume.from_name("retention_data")
    }
)
def entry():
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    import torch
    import os
    import pandas as pd
    import numpy as np
    import time as time
    from tqdm import tqdm
    
    from datetime import datetime
    import shutil
    
    from data_module import PatientDataModule
    from model_module import PatientModelModule 
    from preprocess.const import ROBUST_SCALE_CONST
    
    from train_util.utils import load_hparams_from_yaml
    
    hparams_path = f"/root/configs/{model_name}.yaml"
    hparams = load_hparams_from_yaml(hparams_path)
    
    datamodule = PatientDataModule(hparams_path)
    # model = PatientModelModule(hparams_path)
    
    model = PatientModelModule.load_from_checkpoint(
        checkpoint_path="/root/saved/switch_transformer/2024-12-02T07:46:05.157371_switch_transformer/checkpoints/Switch_Former_epoch(34)_step(1050)_val_0.1495.ckpt",
        hparams_path=hparams_path
    ).eval().cuda()

    df = pd.read_csv("data/unnorm_data_reduced_shift_update_lost.csv")
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
            # print(full_data.columns)
            full_data = full_data.drop(full_data.columns[[0,1,2,3,4,5]], axis=1).drop(["Target","Target_shift_1"]+list(full_data.columns[-11:]),axis=1)
            # print(full_data.columns)
            full_data = full_data.values.astype(float)
            input_data = torch.from_numpy(full_data)
            data.append(input_data)
            count_instances+=1
            
        input_data = torch.stack(data).float().cuda()
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
        
        df.loc[row_index,"mean"] = mean_unlog.cpu().numpy()
        df.loc[row_index,"std"] = std_unlog.cpu().numpy()
        
    print((time.time()-start_time)/count_instances)
        
    df.to_csv("./data/unnorm_data_filtered_lost_mean_std.csv")
