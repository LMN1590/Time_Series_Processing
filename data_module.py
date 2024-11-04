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
SCALE_MEAN = np.array([col["mean"] for col in SCALE_COLS.values()]) + 1e-6
SCALE_STD  = np.array([col["std"] for col in SCALE_COLS.values()]) + 1e-6

class PatientDataset(Dataset):
    def __init__(
        self,
        dataframe:pd.DataFrame,
        pt_idx:List[int],
        output_field:str,
        
        random_sample_from_each_pt:bool = True, equal_sample:bool = True,
        time_steps_len:int = 4, sampled_steps:int = 10
    ):
        self.dataframe = dataframe
        self.output_field = output_field
        self.pt_idx = pt_idx
        
        self.time_steps_len = time_steps_len
        self.sampled_steps= sampled_steps
        self.random_sample_from_each_pt = random_sample_from_each_pt
        self.equal_sample = equal_sample
    
    def __len__(self):
        return len(self.pt_idx)

    def __getitem__(self, index):
        pt_idx = self.pt_idx[index]
        patient_data = self.dataframe.loc[pt_idx].reset_index()
        if self.random_sample_from_each_pt:
            if self.equal_sample:
                sampled_row = patient_data[self.time_steps_len:].sample(n=self.sampled_steps)
            else:
                sampled_row = patient_data[self.time_steps_len:].sample(frac=1.)
            row_index = sampled_row.index.to_list()
        else:
            sampled_upper_lin = self.time_steps_len+self.sampled_steps if self.equal_sample else patient_data.shape[0]
            row_index = list(range(self.time_steps_len,sampled_upper_lin))
        
        data_x = []
        data_y = []
        for idx in row_index:
            serie = list(range(idx-self.time_steps_len,idx))
            try:
                full_data = patient_data.loc[serie]
            except Exception as e:
                print(serie)
                print(idx)
            data_x.append(torch.from_numpy(
                full_data.drop(["PatientId",self.output_field],axis=1)
                .values.astype(float)
            ))
            data_y.append(torch.from_numpy(
                np.asarray(patient_data.loc[idx][self.output_field])
            ))
        return torch.stack(data_x).float(),torch.stack(data_y).float()
        


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
        
        self.dataframe = self.__read_data__(self.file_path)
        self.patient_ids = pd.unique(self.dataframe.index)
        self.data_split_index = self.__get_data_split__(self.patient_ids)
        
        if self.scale:
            self.scaler = StandardScaler(mean = SCALE_MEAN, std = SCALE_STD)
            self.dataframe[list(SCALE_COLS.keys())] = self.scaler.transform(self.dataframe[list(SCALE_COLS.keys())])
            
        
    # region Utils
    def __prepare_params__(
        self, root_path:str,data_path:str = 'unnorm_data.csv',
        output_field:str = 'Target',
        
        random_sample_from_each_pt:bool = True, equal_sample:bool = True,
        time_steps_len:int = 4, min_time_steps:int = 20, sampled_steps:int = 17,
        data_split:List[float] = [0,7,0.1,0.2],
        scale:bool = True,
        
        batch_size:int = 50, num_workers:int = 3,
        drop_last:bool = False, pin_memory:bool=False
    ):
        assert len(data_split)==3
        assert min_time_steps >= sampled_steps+time_steps_len-1, "Minimum Time Steps for each patient need to be higher"
        self.file_path = os.path.join(root_path,data_path)
        self.output_field = output_field
        
        self.time_steps_len = time_steps_len
        self.sampled_steps = sampled_steps
        self.min_time_steps = min_time_steps
        self.random_sample_from_each_pt = random_sample_from_each_pt
        self.equal_sample = equal_sample
        
        self.scale = scale
        
        self.data_split = data_split
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
    def __read_data__(self,file_path:str):
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[[0,1,3]], axis=1)
        
        df = df[df.groupby('PatientId')['PatientId'].transform('size') >= self.min_time_steps]
        
        pt_ids = df["PatientId"].unique()
        # random.shuffle(pt_ids)
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
    # endregion
    
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage):
        self.data_train = PatientDataset(
            dataframe       = self.dataframe.loc[self.data_split_index[0]],
            pt_idx          = self.data_split_index[0],
            output_field    = self.output_field,
            
            random_sample_from_each_pt = self.random_sample_from_each_pt,
            equal_sample=self.equal_sample,
            time_steps_len = self.time_steps_len, 
            sampled_steps = self.sampled_steps
        )
        self.data_val = PatientDataset(
            dataframe       = self.dataframe.loc[self.data_split_index[1]],
            pt_idx          = self.data_split_index[1],
            output_field    = self.output_field,
            
            random_sample_from_each_pt = False,
            equal_sample=self.equal_sample,
            time_steps_len = self.time_steps_len, 
            sampled_steps = self.sampled_steps
        )
        self.data_test = PatientDataset(
            dataframe       = self.dataframe.loc[self.data_split_index[2]],
            pt_idx          = self.data_split_index[2],
            output_field    = self.output_field,
            
            random_sample_from_each_pt = False,
            equal_sample=self.equal_sample,
            time_steps_len = self.time_steps_len, 
            sampled_steps = self.sampled_steps
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            shuffle=False,
        )