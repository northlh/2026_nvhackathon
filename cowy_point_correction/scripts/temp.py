import xarray as xr

file = xr.open_dataset('/project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs/ifs_2024-02-01_00:00:00_f72.nc')
print(file)