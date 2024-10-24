import boto3
import sagemaker
from datetime import datetime
from sagemaker.pytorch import PyTorch,PyTorchModel

sagemaker_session_bucket = "sgn-sagemaker-dev"
sess = sagemaker.Session(
    default_bucket=sagemaker_session_bucket,
    boto_session=boto3.Session(region_name = 'us-west-2')
)

if(sagemaker_session_bucket is None and sess is not None):
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole-20240730T174360')['Role']['Arn']
print(role)
        
weight_name = "PatientRetention--2024-10-23T17-27-49610547"
model = PyTorchModel(
    model_data=f"s3://sgn-sagemaker-dev/patient_retention/model_training_logs/{weight_name}/output/model.tar.gz",
    role=role,
    entry_point="inference.py",

    framework_version='2.3.0',          # PyTorch version
    py_version='py311',  
    
    name = weight_name,
    source_dir = "import_module"
)

predictor = model.deploy(
    initial_instance_count= 1,
    instance_type = 'ml.g4dn.xlarge',
    model_name = weight_name,
    endpoint_name = weight_name
)

# predictor.delete_endpoint()
# predictor.delete_model()