
This is my part of 4th Place Solution for the Child Mind Institute - Detect Sleep States (Kaggle competition).

detail document: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459597


# How to Reproduce (for Competition Organizers)
## Hardware
- CPU: Intel Core i9 13900KF (24 cores, 32 threads)
- GPU: NVIDIA GeForce RTX 4090
- RAM: 64GB

## OS/platform
- WSL2 (version 2.0.9.0, Ubuntu 22.04.2 LTS)

## 3rd-party software
Please check the dockerfile in `/kaggle/.devcontainer`

## Training
1. Upload competition dataset in `/kaggle/input/child-mind-institute-detect-sleep-states`
    - i.e. `/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet`, etc...
2. Run following notebooks to prepare input dataset:
    - `/kaggle/input/cv-split/split.ipynb`
    - `/kaggle/input/heauristic_features/heuristic.ipynb`
    - `/kaggle/input/save_series_csv/save_gaussian.ipynb`
3. Run follwing notebooks to train models:
    - `/kaggle/train/patch_transformer_gru.ipynb`
    - `/kaggle/train/patch_transformer_gru_v2.ipynb`
    - `/kaggle/train/patch_wavenet_gru.ipynb`
    - `/kaggle/train/patch_1dcnn_gru.ipynb`
    - `/kaggle/train/patch_transformer_gru_exp133.ipynb` 
    - `/kaggle/train/next_cands.ipynb`
    - `/kaggle/train/xgboost.ipynb`
    - `/kaggle/train/lightgbm.ipynb`


NOTE:
- Directories (`/kaggle/output/exp_XXX`) are created and the output files (including model weights) are stored in them.
- To avoid OOM, free memory when each notebook is finished executing.
- The training codes will delete all data in `/kaggle/working` when they are executed.

## Supplemental Information for Competition Organizers
- Dockerfile is used instead of `B4.requirements.txt`.
- `src/config.yaml` is used instead of `B6. SETTINGS.json`.
- `B7. Serialized copy of the trained model` is [here](https://www.kaggle.com/datasets/ryotayoshinobu/zzz-output).
- `B8. entry_points.md` is not included because my all codes are `.ipynb` format.

