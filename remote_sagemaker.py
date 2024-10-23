import boto3
import sagemaker
from datetime import datetime

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
    
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role = role,
    instance_count = 1,
    instance_type = 'ml.g4dn.xlarge',
    framework_version='2.3.0',          # PyTorch version
    py_version='py311',  
    input_mode='File', 
    dependencies=['import/','data_module.py','model_module.py','sagemaker-requirements.txt','const.py'],
    output_path="s3://sgn-sagemaker-dev/patient_retention/model_training_logs/",
    code_location="s3://sgn-sagemaker-dev/patient_retention/model_training_logs/"
)
estimator.fit(
    's3://sgn-sagemaker-dev/patient_retention/data',
    job_name=f'PatientRetention--{datetime.now().isoformat().replace(":","-").replace(".","")}'
)