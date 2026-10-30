# Deep learning-based cancer detection on PatchCamelyon (PCam)

Code for the LEEN364 project *"Deep Learning-Based Cancer Detection in Histopathology Images:
A Comparison of CNN Architectures and Transfer Learning on the PatchCamelyon Dataset"*
(Başak Erdağ, 240303907).

The project classifies 96 x 96 histopathology patches as metastatic / non-metastatic and compares
Logistic Regression, an MLP, a small CNN, ImageNet-pretrained ResNet18 and EfficientNet-B0, data augmentation,
activation functions and optimizers, and autoencoder pre-training versus ImageNet transfer learning with
1% and 10% of the labels. Interpretability is studied with Grad-CAM.

## Data

The official PCam split (262,144 / 32,768 / 32,768 patches) is used. The reported experiments used the Kaggle copy
<https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon>
(original data: <https://github.com/basveeling/pcam>).

`src/data.py` detects the Kaggle folder automatically. For a local run, place the original files
`camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5` (decompressed) in `data/`.

## Setup

```
pip install -r requirements.txt
```

`results/dataset_stats.json` (per-channel mean / std over all 262,144 training patches, used for standardization) is created by `src/eda.py`, which reads the original files from `data/` (run it locally once: `python -m src.eda`).

## Code structure

| File | Purpose |
|---|---|
| `src/data.py` | Loads a PCam split into RAM (uint8) |
| `src/models.py` | Logistic Regression, MLP, small CNN, ResNet18 / EfficientNet-B0 wrappers, convolutional autoencoder |
| `src/train.py` | Training / evaluation of one run: augmentation, early stopping, threshold selection on validation, single test evaluation |
| `src/pretrain_ae.py` | Unsupervised autoencoder pre-training |
| `src/metrics.py` | Validation threshold selection and test metrics |
| `src/utils.py`, `src/eda.py` | Seeding / device, exploratory analysis |
| `scripts/eda_report.py` | Class distribution, sample patches and colour histograms shown in the report |
| `scripts/` (others) | Aggregation of the JSON results into tables and figures |

Every run of `src.train` writes `results/<tag>.json` (arguments, validation and test metrics, threshold, per-epoch history),
`results/preds/<tag>.npz` (probabilities) and `results/ckpt/<tag>.pt` (best weights).

## Commands used for the experiments

`S` is the seed (0, 1, 2). Runs were distributed over two GPUs with `CUDA_VISIBLE_DEVICES`.

```
# baselines
python -m src.train --model logreg --epochs 10 --seed S
python -m src.train --model mlp --epochs 10 --seed S

# augmentation (small CNN, ReLU, Adam 1e-3)
python -m src.train --model smallcnn --aug none|geo|geo_color --epochs 10 --seed S

# learning-rate selection for the ablation (ReLU, seed 0, 50k subset): Adam 1e-3, 3e-4; SGD 1e-2, 1e-1
python -m src.train --model smallcnn --aug geo --opt adam|sgd --lr LR --epochs 10 --seed 0 --subset 50000

# ablation (selected: Adam 1e-3, SGD 0.1)
python -m src.train --model smallcnn --aug geo --act relu|leakyrelu|gelu --opt adam --lr 1e-3 --epochs 10 --seed S --subset 50000
python -m src.train --model smallcnn --aug geo --act relu|leakyrelu|gelu --opt sgd  --lr 0.1  --epochs 10 --seed S --subset 50000

# transfer learning (epoch 1: frozen backbone)
python -m src.train --model resnet18|efficientnet_b0 --aug geo --norm imagenet --freeze 1 --lr 1e-4 --lr_head 1e-3 --epochs 6 --seed S

# autoencoder pre-training (one per seed)
python -m src.pretrain_ae --seed S --epochs 8

# low-label experiment (F = 0.01 with 40 epochs, or 0.10 with 20 epochs; patience 5)
python -m src.train --model smallcnn --aug geo --label_frac F --epochs E --patience 5 --lr 1e-3 --seed S --suffix _lfXX_scratch
python -m src.train --model smallcnn --aug geo --label_frac F --epochs E --patience 5 --lr 1e-3 --seed S --init_ae results/ae/ae_seedS_encoder.pt --suffix _lfXX_ae
python -m src.train --model resnet18 --aug geo --norm imagenet --freeze 1 --lr 1e-4 --lr_head 1e-3 --label_frac F --epochs E --patience 5 --seed S --suffix _lfXX_imagenet
```

## Notes

* The decision threshold is chosen on the validation set (balanced accuracy); the test set is evaluated once per run.
* cuDNN autotuning is enabled for speed, so runs are seeded but not guaranteed to be bit-wise reproducible.
* Console messages and some comments in the code are in Turkish.
