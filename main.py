from data_module import Dataset_MTS
import numpy

ds = Dataset_MTS(
    root_path="data/",
    in_len = 16
)
print(type(ds[2][0]))
print(ds[2][1].dtype)