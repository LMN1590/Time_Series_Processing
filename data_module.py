from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader

import random
from typing import List
import pandas as pd
import os

from train_util.utils import load_hparams_from_yaml,StandardScaler


class Dataset_MTS(Dataset):
    def __init__(
        self, root_path, data_path='ETTh1.csv', 
        output_field:str = 'OT',
        flag='train', in_len=16, 
        data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.output_field = output_field
        self.in_len = in_len
        self.out_len = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        #self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(
            self.root_path,
            self.data_path
        ))
        
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]
        else:
            train_num = int(len(df_raw)*self.data_split[0])
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num
        
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        self.cols_idx = cols_data.get_loc(self.output_field)

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler(cols_idx=self.cols_idx)
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(
                    mean = self.scale_statistic['mean'], 
                    std = self.scale_statistic['std'],
                    cols_idx=self.cols_idx
                )
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end][:,self.cols_idx]

        return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float()
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
    

class TestDataset(Dataset):
    def __init__(
        self,
        random_idx:List[int],
        
        input_dim:int,
        output_dim:int,
        time_steps:int
    ):
        super().__init__()
        self.random_idx = random_idx
        self.time_steps = time_steps
        self.input_dim  = input_dim
        self.output_dim = output_dim
        
    def __len__(self):
        return 5000
    def __getitem__(self,idx:int):
        input = torch.randn((self.input_dim*self.time_steps))
        output = torch.randn((self.output_dim))
        
        input[self.random_idx] = output
        input = input.reshape((self.time_steps,-1))
        
        return input, output
    

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
        
        
        self.data_train = Dataset_MTS(
            root_path=self.data_param["root_path"],
            data_path=self.data_param["data_path"],
            output_field=self.data_param['output_field'],
            flag='train',
            in_len=self.data_param["in_len"],
            data_split=self.data_param["data_split"],
            scale=True
        )
        self.data_val   = Dataset_MTS(
            root_path=self.data_param["root_path"],
            data_path=self.data_param["data_path"],
            output_field=self.data_param['output_field'],
            flag='val',
            in_len=self.data_param["in_len"],
            data_split=self.data_param["data_split"],
            scale=True
        )
        self.data_test  = Dataset_MTS(
            root_path=self.data_param["root_path"],
            data_path=self.data_param["data_path"],
            output_field=self.data_param['output_field'],
            flag='test',
            in_len=self.data_param["in_len"],
            data_split=self.data_param["data_split"],
            scale=True
        )
        
        self.output_idx = self.data_train.cols_idx
        self.scaler_params = {
            "mean":self.data_train.scaler.mean.tolist(),
            "std":self.data_train.scaler.std.tolist()
        }
        
    def prepare_data(self):
        return super().prepare_data()
    
    def setup(self, stage):
        return super().setup(stage)

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