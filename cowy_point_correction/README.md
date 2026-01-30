
# CoWy Point Correction

Production-grade PyTorch Lightning pipeline for point-based
bias correction of HRRR / IFS wind forecasts using MADIS observations.

## Typical Usage
### prepare data usage arguments are configs, terrain path, and ifs path
### train the model
### evaluate the models


### run this script without the arguments or it will copy the dataset
```bash

python scripts/prepare_data.py configs/ifs_v1.yaml \
  --terrain-src /project/cowy-nvhackathon/cowy-wildfire/data/terrain_data/terrain_990m/ \
  --ifs-src /project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs/
```
###data for the model needs to also be prepared. The train.py has a prepare mode that can be activated using 
```bash
python scripts/train.py configs/ifs_v1.yaml --prepare-only
```
### we can then run the training with the data as
```bash
python scriptsd/train.py configs/ifs_v1.yaml
```bash
python scripts/evaluate.py configs/ifs_v1.yaml
```
# to change which subset of data go to traning/datamodule.py change everything in the bracket
```python
 ### change to subset
        hrrr_fps = sorted(glob.glob(os.path.join(ifs_dir, "*.nc")))[:20]
```
### the self check script is to examine the structure of the data to run it 

```bash
# Just HRRR/IFS
python scripts/self_check_script.py --hrrr /project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs/ifs_2024-12-31_00:00:00_f81.nc

# With MADIS (tests KDTree NN using obs coords)
python scripts/self_check_script.py --hrrr /path/to/hrrr.nc --madis /path/to/madis.nc

# With topo multi-file open test (silences compat FutureWarning)
python scripts/self_check_script.py --hrrr /path/to/hrrr.nc --topo "/path/to/topo/*.nc"
```
