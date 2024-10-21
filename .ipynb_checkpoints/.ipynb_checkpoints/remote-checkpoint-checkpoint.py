from modal import App,Image,gpu,Mount,Volume

app = App("testing")

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg","libsm6","libxext6"])
    .pip_install_from_requirements("./requirements.txt")
)

@app.function(
    image=image,
    gpu = "t4",
    timeout=86400,
    retries=0,
    mounts = [
        Mount.from_local_dir("configs",remote_path="/root/configs"),
        Mount.from_local_dir("model",remote_path="/root/model"),
        Mount.from_local_dir("train_util",remote_path="/root/train_util"),
        Mount.from_local_dir("data",remote_path="/root/data"),
        
        Mount.from_local_file("const.py","/root/const.py"),
        Mount.from_local_file("data_module.py","/root/data_module.py"),
        Mount.from_local_file("main.py","/root/main.py"),
        Mount.from_local_file("model_module.py","/root/model_module.py"),
        Mount.from_local_file("remote.py","/root/remote.py")
    ],
    volumes={
        "/root/saved": Volume.from_name("retention_log")
    }
)
def entry():
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
    from pytorch_lightning.loggers import CSVLogger
    import torch
    
    from data_module import PatientDataModule
    from model_module import PatientModelModule 
    
    from train_util.utils import load_hparams_from_yaml
    
    hparams_path = "/root/configs/crossformer.yaml"
    hparams = load_hparams_from_yaml(hparams_path)
    datamodule = PatientDataModule(hparams_path)
    model = PatientModelModule(hparams_path)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="/root/saved/checkpoints",
        filename=f'{hparams["model"]["name"]}' + "_epoch({epoch:02d})_step({step:04d})_val_{val/mse:.4f}",
        
        monitor="val/mse",
        mode="min",
        
        auto_insert_metric_name=False,
        
        save_last=True,
        save_weights_only=True,
        save_top_k=3,
    )
    csv_logger = CSVLogger(
        save_dir="/root/saved/logs"
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