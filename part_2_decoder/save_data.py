import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client


def main():
    client = Client(n_workers=8, threads_per_worker=2)
    print("Dashboard:", client.dashboard_link)
    target_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    ds_full = xr.open_dataset(
        target_path,
        engine="zarr",
        chunks={},
        backend_kwargs={"storage_options": {"anon": True}},
    )
    ds_full = ds_full.resample(time="1D").mean()
    n_samples = len(ds_full.time.values)
    print(n_samples)
    ds_full = xr.Dataset(
        {
            "2m_temperature": ds_full["2m_temperature"],
            "u_component_of_wind": ds_full["u_component_of_wind"].sel(
                level=850, drop=True
            ),
            "v_component_of_wind": ds_full["v_component_of_wind"].sel(
                level=850, drop=True
            ),
            "geopotential": ds_full["geopotential"].sel(level=850, drop=True),
            "specific_humidity": ds_full["specific_humidity"].sel(
                level=850, drop=True
            ),
        }
    )

    ds_full = ds_full.to_stacked_array(
        new_dim="batch", sample_dims=("time", "longitude", "latitude")
    ).transpose("time", "batch", ...)

    print("HERE")
    with ProgressBar():
        tensor_data = ds_full.values
        print(tensor_data.shape)
        tensor_data = torch.from_numpy(tensor_data).float()
        torch.save(
            tensor_data, "/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt"
        )


if __name__ == "__main__":
    main()
