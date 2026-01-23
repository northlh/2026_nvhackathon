
# CoWy Point Correction

Production-grade PyTorch Lightning pipeline for point-based
bias correction of HRRR / IFS wind forecasts using MADIS observations.

## Typical Usage

```bash
python scripts/prepare_data.py configs/ifs_v1.yaml
python scripts/train.py configs/ifs_v1.yaml
python scripts/evaluate.py configs/ifs_v1.yaml
