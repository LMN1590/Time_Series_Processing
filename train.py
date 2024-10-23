import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r" ,package])
install("sagemaker-requirements.txt")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from data_module import PatientDataModule
from model_module import PatientModelModule 

from train_util.utils import load_hparams_from_yaml

OUTPUT_DIR = '/opt/ml/model'

hparams_path = "configs/rnn.yaml"
hparams = load_hparams_from_yaml(hparams_path)
datamodule = PatientDataModule(hparams_path)
model = PatientModelModule(hparams_path)

os.makedirs(os.path.join(OUTPUT_DIR,"checkpoints"))
os.makedirs(os.path.join(OUTPUT_DIR,"logs"))
checkpoint_cb = ModelCheckpoint(
    dirpath=os.path.join(OUTPUT_DIR,"checkpoints"),
    filename=f'{hparams["model"]["name"]}' + "_epoch({epoch:02d})_step({step:04d})_val_{val/mse:.4f}",
    
    monitor="val/mse",
    mode="min",
    
    auto_insert_metric_name=False,
    
    save_last=True,
    save_weights_only=True,
    save_top_k=1,
)
csv_logger = CSVLogger(
    save_dir=os.path.join(OUTPUT_DIR,"logs")
)
early_stopping_cb = EarlyStopping(
    monitor="val/mse",
    mode="min",
    patience=5,
    min_delta = 0
)

# input_temp = torch.randn((4,16,49))
# output = model.net(input_temp)
# print(output.shape)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    logger=csv_logger,
    precision=32,
    max_epochs=hparams["training_param"]["max_epochs"],
    check_val_every_n_epoch=1,
    callbacks=[
        checkpoint_cb,
        early_stopping_cb
    ],
    log_every_n_steps=5
)

trainer.fit(model=model,datamodule=datamodule)
trainer.test(model=model,datamodule=datamodule,ckpt_path="last")
trainer.test(model=model,datamodule=datamodule,ckpt_path="best")

print(os.listdir())
print()
print(os.listdir("/opt/ml/input"))
print()
print(os.listdir("/opt/ml/input/data/training"))
print()
print(os.listdir("/opt/ml/output"))
print()
print(os.listdir("/opt/ml/model"))
print()