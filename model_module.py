import yaml
from typing import Tuple
import time

from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

from const import MODEL_LIST,LOSS_LIST
from train_util.lr_scheduler import LinearWarmupCosineAnnealingLR
from train_util.utils import load_hparams_from_yaml
from preprocess.const import ROBUST_SCALE_CONST,STANDARD_SCALE_CONST,ROBUST_SCALE_NO_LOG

class PatientModelModule(LightningModule):    
    def __init__(
        self,
        hparams_path:str
    ):
        super(PatientModelModule,self).__init__()
        self.save_hyperparameters()
        
        hparams = load_hparams_from_yaml(hparams_path)
        self.hyperparameters = hparams
        
        model_params = hparams["model"]
        self.net = MODEL_LIST[model_params["name"]](
            **model_params["params"]
        )
        
        loss_params = hparams["loss"]
        self.loss_params = loss_params
        for loss in loss_params:
            setattr(
                self,
                loss["name"],
                LOSS_LIST[loss["name"]](
                    **loss["params"]
                )
            )
        self.denorm_loss = nn.MSELoss()
    
    def training_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        var = prediction[1]
        
        loss = torch.zeros(1).float().cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,var,output) * loss_setting["weight"]
            loss += loss_val
            self.log(
                f"train/{loss_setting['name']}", loss_val,prog_bar=True,logger=True
            )

        self.log(
            "train/all", loss, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        var = prediction[1]
        
        loss = torch.zeros(1).float().cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,var,output) * loss_setting["weight"]
            loss += loss_val
            self.log(
                f"val/{loss_setting['name']}", loss_val,prog_bar=True,logger=True
            )
        
        self.log(
            "val/all", loss, prog_bar=True, logger=True
        )
        return loss
    
    def test_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        var = prediction[1]
        
        loss = torch.zeros(1).float().cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,var,output) * loss_setting["weight"]
            loss += loss_val
            self.log(
                f"test/{loss_setting['name']}", loss_val,prog_bar=True,logger=True
            )
        
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
        output_unnorm = output*scale+center
    
        mean_unlog = torch.exp(mean_unnorm + (var_unnorm)/2)-1
        output_unlog = torch.exp(output_unnorm)-1
        denorm_loss = self.denorm_loss(mean_unlog,output_unlog)
        
        
        self.log(
            "test/all", loss, prog_bar=True, logger=True
        )
        self.log(
            "test/pred_mean", mean_unlog.mean().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/pred_max", mean_unlog.max().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/pred_min", mean_unlog.min().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/pred_std", mean_unlog.std().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/output_mean", output_unlog.mean().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/output_max", output_unlog.max().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/output_min", output_unlog.min().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/output_std", output_unlog.std().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/pred_denorm_mse", torch.sqrt(denorm_loss),prog_bar=True,logger=True
        )
        return loss
    
    def forward(self,x):
        prediction = self.net(x)
        return prediction

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)
        
        training_param = self.hyperparameters["training_param"]
        optimizer_name = training_param["optimizer"]
        optimizer_param = training_param["params"]
        
        if(optimizer_name == "AdamW"):
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": decay,
                        "lr": optimizer_param["lr"],
                        "betas": (optimizer_param["beta_1"], optimizer_param["beta_2"]),
                        "weight_decay": optimizer_param["weight_decay"],
                    },
                    {
                        "params": no_decay,
                        "lr": optimizer_param["lr"],
                        "betas": (optimizer_param["beta_1"], optimizer_param["beta_2"]),
                        "weight_decay": 0,
                    },
                ]
            )
        elif(optimizer_name == "SGD"):
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": decay,
                        "lr": optimizer_param["lr"],
                        "weight_decay": optimizer_param["weight_decay"],
                        "momentum": optimizer_param["momentum"],
                    },
                    {
                        "params": no_decay,
                        "lr": optimizer_param["lr"],
                        "weight_decay": 0,
                        "momentum": optimizer_param["momentum"], 
                    },
                ]
            )
        
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            training_param["warmup_epochs"],
            training_param["max_epochs"],
            training_param["warmup_start_lr"],
            training_param["eta_min"],
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"lr_scheduler": scheduler, "optimizer": optimizer}