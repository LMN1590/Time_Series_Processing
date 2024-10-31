from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
from typing import List
import pandas as pd
import os
from typing import List

from train_util.utils import load_hparams_from_yaml,StandardScaler
from const import SCALE_COLS
SCALE_MEAN = np.array([col["mean"] for col in SCALE_COLS.values()])
SCALE_STD  = np.array([col["std"] for col in SCALE_COLS.values()])

class PatientDataset(Dataset):
    def __init__(
        self, root_path,data_path = 'unnorm_data.csv',
        output_field:str = 'Target',
        random_sample_from_each_pt:bool = True,
        flag:str = 'train', 
        time_steps_len:int = 4, min_time_steps:int = 20,
        data_split:List[float] = [0,7,0.1,0.2],
        scale:bool = True, scale_statistic=None
    ):
        assert flag in ['train', 'test', 'val']
        assert time_steps_len <= min_time_steps, f"time_steps_len should be smaller than min_time_steps - {min_time_steps}"
        
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.data_split = data_split
        
        self.output_field = output_field
        self.time_steps_len = time_steps_len
        self.min_time_steps = min_time_steps
        self.random_sample_from_each_pt = random_sample_from_each_pt
        
        self.scale = scale

        self.dataframe = self.__read_data__(root_path,data_path) 
        self.patient_ids = pd.unique(self.dataframe.index)
        self.total_valid_records = len(self.dataframe)


class PatientDataModule(LightningDataModule):
    def __init__(
        self,
        hparams_path:str
    ):
        super().__init__()
        self.save_hyperparameters()
        
        hparams = load_hparams_from_yaml(hparams_path)
        self.hyperparameters = hparams
        self.data_param = hparams["data"]
        
        self.__prepare_params__(**self.data_param)
        if self.scale:
            self.scaler = StandardScaler(mean = SCALE_MEAN, std = SCALE_STD)
        
        
    def __prepare_params__(
        self, root_path:str,data_path:str = 'unnorm_data.csv',
        output_field:str = 'Target',
        random_sample_from_each_pt:bool = True,
        time_steps_len:int = 4, min_time_steps:int = 20,
        data_split:List[float] = [0,7,0.1,0.2],
        scale:bool = True,
        batch_size:int = 50, num_workers:int = 3,
        drop_last:bool = False, pin_memory:bool=False
    ):
        self.file_path = os.path.join(root_path,data_path)
        self.output_field = output_field
        
        self.time_steps_len = time_steps_len
        self.min_time_steps = min_time_steps
        self.random_sample_from_each_pt = random_sample_from_each_pt
        
        self.scale = scale
        
        self.data_split = data_split
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
    def __read_data__(self,file_path:str):
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[0],axis=1)
        
        df = df[df.groupby('PatientId')['PatientId'].transform('size') >= self.min_time_steps]
        
        pt_ids = df["PatientId"].unique()
        random.shuffle(pt_ids)
        df = df.set_index("PatientId").loc[pt_ids]
        
        return df
    def __get_data_split__(self,patient_ids:np.ndarray):
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]
        else:
            train_num = int(len(self.patient_ids)*self.data_split[0])
            test_num = int(len(self.patient_ids)*self.data_split[2])
            val_num = len(self.patient_ids) - train_num - test_num
        
        border1s = [0, train_num, train_num + val_num]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
        
        return [patient_ids[borders[0]:borders[1]] for borders in zip(border1s,border2s)]
        
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage):
        self.dataframe = self.__read_data__(self.file_path)
        self.patient_ids = pd.unique(self.dataframe.index)
        
        self.data_split_index = self.__get_data_split__(self.patient_ids)
        if self.scale:
            self.dataframe[SCALE_COLS.keys()] = self.scaler.transform(self.dataframe[SCALE_COLS.keys()])
        
        
        self.data_train = PatientDataset(
            dataframe       = self.dataframe,
            patient_ids     = self.data_split_index[0],
            output_field    = self.output_field
        )


    

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.data_param["batch_size"],
            num_workers=self.data_param["num_workers"],
            drop_last=self.data_param["drop_last"],
            pin_memory=self.data_param["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.data_param["batch_size"],
            num_workers=self.data_param["num_workers"],
            drop_last=self.data_param["drop_last"],
            pin_memory=self.data_param["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.data_param["batch_size"],
            num_workers=self.data_param["num_workers"],
            drop_last=self.data_param["drop_last"],
            pin_memory=self.data_param["pin_memory"],
            shuffle=False,
        )