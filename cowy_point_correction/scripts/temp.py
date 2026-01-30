import xarray as xr
import glob, os

fp = sorted(glob.glob("/project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs_small/*.nc"))[0]
ds = xr.open_dataset(fp)

vars_to_check = ["u10","v10","ws_10","t2m","sp","t_500hPa","elr"]
for v in vars_to_check:
    if v in ds:
        print(v, ds[v].dims, ds[v].shape)
    else:
        print(v, "MISSING")