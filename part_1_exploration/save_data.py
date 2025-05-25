import numpy as np
import pandas as pd
import torch
import xarray as xr

target_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
variables = [
    "2m_temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]


ds_full = xr.open_dataset(
    target_path,
    engine="zarr",
    chunks={},
    backend_kwargs={"storage_options": {"anon": True}},
)[variables]

ds_full = ds_full.resample(time="1D").mean().isel(level=0)
ds_full = ds_full.to_stacked_array(
    new_dim="batch", sample_dims=("time", "longitude", "latitude")
).transpose("time", "batch", ...)
print(ds_full)

timestamps = ds_full.time.values
print(timestamps)
Y = []
for timestamp in timestamps:
    month = pd.Timestamp(timestamp).month
    if month in [12, 1, 2]:
        Y.append(0)
    elif month in [3, 4, 5]:
        Y.append(1)
    elif month in [6, 7, 8]:
        Y.append(2)
    elif month in [9, 10, 11]:
        Y.append(3)
Y = torch.tensor(Y)
torch.save(Y, "/vol/bitbucket/nb324/era5_level0_Y.pt")
ds_full = ds_full.values
np.save("/vol/bitbucket/nb324/era5_level0.npy", ds_full)
tensor_data = torch.from_numpy(ds_full).float()
torch.save(tensor_data, "/vol/bitbucket/nb324/era5_level0.pt")
