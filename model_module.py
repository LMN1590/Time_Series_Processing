import yaml
from typing import Tuple

from pytorch_lightning import LightningModule
import torch

from const import MODEL_LIST,LOSS_LIST
from train_util.lr_scheduler import LinearWarmupCosineAnnealingLR
from train_util.utils import load_hparams_from_yaml

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
    
    def training_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        std = prediction[1]
        
        loss = torch.zeros(1).cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,std,output) * loss_setting["weight"]
            loss += loss_val

        self.log(
            "train/mse", loss, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        std = prediction[1]
        
        loss = torch.zeros(1).cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,std,output) * loss_setting["weight"]
            loss += loss_val
        
        self.log(
            "val/mse", loss, prog_bar=True, logger=True
        )
        return loss
    
    def test_step(self,batch:Tuple[torch.Tensor, torch.Tensor],batch_idx):
        input, output = batch
        input = input.flatten(0,1)
        output = output.flatten(0,1)
        
        prediction = self.net(input)
        mean = prediction[0]
        std = prediction[1]
        
        loss = torch.zeros(1).cuda()
        
        for idx, loss_setting in enumerate(self.loss_params):
            loss_func = getattr(self,loss_setting["name"])
            loss_val = loss_func(mean,std,output) * loss_setting["weight"]
            loss += loss_val
        
        self.log(
            "test/mse", loss, prog_bar=True, logger=True
        )
        self.log(
            "test/pred_mean", mean.mean().item(),prog_bar=True,logger=True
        )
        self.log(
            "test/pred_std", std.mean().item(),prog_bar=True,logger=True
        )
        return loss

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