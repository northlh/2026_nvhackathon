
# CoWy Point Correction

Production-grade PyTorch Lightning pipeline for point-based
bias correction of HRRR / IFS wind forecasts using MADIS observations.

## Typical Usage
### prepare data usage arguments are configs, terrain path, and ifs path
### train the model
### evaluate the models
```bash

python scripts/prepare_data.py configs/ifs_v1.yaml \
  --terrain-src /projects/cowy/datasets/terrain_data/terrain_990m/ \
  --ifs-src /scratch/kylabazlen/herbie/ifs/
python scripts/train.py configs/ifs_v1.yaml
python scripts/evaluate.py configs/ifs_v1.yaml
