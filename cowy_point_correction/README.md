
# CoWy Point Correction

Production-grade PyTorch Lightning pipeline for point-based
bias correction of HRRR / IFS wind forecasts using MADIS observations.

## Typical Usage
### prepare data usage arguments are configs, terrain path, and ifs path
### train the model
### evaluate the models
```bash

python scripts/prepare_data.py configs/ifs_v1.yaml \
  --terrain-src /project/cowy-nvhackathon/cowy-wildfire/data/terrain_data/terrain_990m/ \
  --ifs-src /project/cowy-nvhackathon/cowy-wildfire/data/nwp/ifs/
python scripts/train.py configs/ifs_v1.yaml
python scripts/evaluate.py configs/ifs_v1.yaml
