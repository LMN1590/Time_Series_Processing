import subprocess
import sys
import os
import json

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r" ,package])


import torch

print("Current Dir")
cur_dir = os.getcwd()
print(cur_dir)
print(os.listdir(cur_dir))
print("Code Dir")
code_dir = os.path.join(cur_dir,"code")
print(code_dir)
print(os.listdir(code_dir))
install(os.path.join(code_dir,"sagemaker-requirements.txt"))
    
def model_fn(model_dir):
    print("Get current model_dir stuff")
    print(model_dir)
    print(os.listdir(model_dir))
    
    sys.path.append("./code")
    from model_module import PatientModelModule 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = os.path.join(model_dir,"checkpoints")
    ckpt_path = [path for path in os.listdir(ckpt_dir) if path.endswith(".ckpt") and path != "last.ckpt"][0]
    model = PatientModelModule.load_from_checkpoint(os.path.join(ckpt_dir,ckpt_path))
    model = model.eval().to(device)
    return model

def predict_fn(input_data,model):
    input_tensor = torch.tensor(input_data).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model.predict(input_tensor)
        print(output_tensor.shape)
        return output_tensor.flatten(0).tolist()
    
def input_fn(request_body, content_type):
    if content_type == 'application/json':
        body = json.loads(request_body)
        input_data = torch.tensor(body["input"])
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")