# Modeling — yam canopy cover (U-Net + seeded watershed)

These scripts estimate per-plot canopy area (m²) from RGB drone orthophotos. The steps are:

1. A U-Net (ResNet34 encoder) predicts a canopy probability map over the whole orthophoto.
2. A seeded watershed assigns the canopy to each plot, including canopy that grows past the plot box, and splits plants that touch.

The model in `Weight/` was trained on 104 hand-annotated RoujolA 2024-12-12 plots.

```
Modeling/
├── environment.yml          conda env (yam-fcover)
├── Weight/                  unet_canopy_full.pt  — the 104-plot model used by all predict scripts
├── Train/                   add annotations and retrain
│   ├── 1_build_training_data.py
│   ├── 2_train_unet_5fold.py
│   ├── training_data_104/   images/<stem>.tif + masks/<stem>_mask.png  (104 plots)
│   └── results_104/         5-fold CV results of the current model (the baseline)
├── Test/                    run the model on selected site/dates
│   ├── predict_ortho_104.py       step 1: canopy probability map + per-plot area inside the box
│   └── predict_watershed_104.py   step 2: watershed per-plot canopy area
├── Full_canopy_Estimate/    run every site/date listed in the master workbook
│   ├── full_predict_ortho.py      step 1
│   └── full_predict_watershed.py  step 2 → Estimates/all_sites_all_dates.csv
└── Clipping/                needed by all predict scripts (Godet25-26 row/col from the field map)
    ├── current-reclip_batch.py
    └── Godet25_map.csv
```

## Setup

From the repo folder:

```bash
conda env create -f Modeling/environment.yml
conda activate yam-fcover
```

If a `yam-fcover` env already exists, skip `conda env create` and only activate it.

**Data that is not in the repo** (the orthophotos, GRID shapefiles and the `1RGB_MasterDataset_Split_By_Site.xlsx` Site/Date workbook) is read from the Shakoor Lab OneDrive. Every script has a `ROOT` line in its CONFIG block. The default is:

```
~/Library/CloudStorage/OneDrive-DDPSC/Shakoor Lab (Notes)/Yam/DroneData/Orthophotos
```

If your OneDrive is in a different location, change `ROOT`. Cloud-only OneDrive files can cause timeouts or permission errors. If this happens, set the folder to "Always Keep on This Device" and run again.

Every script finds the repo files from its own location, so you can run it from any folder: with the VS Code ▶ button (select the `yam-fcover` interpreter first) or from a terminal:

```bash
conda activate yam-fcover
cd Modeling/Test
python predict_ortho_104.py
python predict_watershed_104.py
```

## Running predictions

| Folder | Which site/dates | Outputs (not committed) |
|---|---|---|
| `Test/` | `SITE_DATES` in the CONFIG block | `Test/Predictions/`, `Test/Watershed/` |
| `Full_canopy_Estimate/` | every Site/Date in the master workbook (`USE_XLSX_DATES = True`) | `Full_canopy_Estimate/Canopy_Probability/`, `Full_canopy_Estimate/Estimates/` |

Always run step 1 (ortho) before step 2 (watershed). The watershed step reads the `canopy_prob.tif` written by step 1 and does not run the model again. Use `NEGATE` in the CONFIG block to skip a site/date.

**Note: `Clipping/Godet25_map.csv` is required for Godet25-26.** The Godet25-26 GRID shapefiles have only a plot `id`, with no row or col. For sites in `ID_ONLY_SITES`, all 4 predict scripts rebuild row/col from the plot positions and this 25 × 33 field map, using the functions in `Clipping/current-reclip_batch.py`. The map supplies only the row/col position. The genotype label always comes from the shapefile `id`. If the map is missing, every Godet25-26 run fails. If you do not process Godet25-26, add `("Godet25-26", "*")` to `NEGATE`.

## Improving the model (new annotations)

1. Annotate new plot clips in Label Studio with the brush label `canopy`, then export JSON.
   - **Annotate dense plots first.** Most of the remaining error is in high-cover plots.
   - Follow one labelling rule for all annotators. Annotator differences were a measurable source of bias.
2. In `Train/1_build_training_data.py`, set `JSON_FILES` (the export) and `CLIP_DIRS` (the folders holding the clip `.tif` files). Then run the script. It copies `training_data_104` unchanged, adds the new plots, and writes `Train/training_data_new/`. Check `Train/training_data_new/_overlays/` by eye.
3. In `Train/2_train_unet_5fold.py`, set `DATA_DIR` to the new folder and give `OUT_DIR` a new name. Then run it (150 epochs × 5 folds, plus a full model).
4. Compare the new `RESULTS_INTERPRETATION.txt` and `fold_summary.csv` (in `OUT_DIR`) against `Train/results_104/`. Replace `Weight/unet_canopy_full.pt` only if the new model is better.

Run steps 2 and 3 with the ▶ button or from a terminal:

```bash
conda activate yam-fcover
cd Modeling/Train
python 1_build_training_data.py
python 2_train_unet_5fold.py
```

**Note: `Train/results_104/` is the baseline.** `2_train_unet_5fold.py` writes `val_report_cv.csv`, `fold_summary.csv` and `RESULTS_INTERPRETATION.txt` to its `OUT_DIR` at the end of every run. `results_104/` holds these files from the run that produced `Weight/unet_canopy_full.pt`, so a new model can be compared without first retraining the 104 model. Do not overwrite it; write new runs to a new `OUT_DIR`. A run with more plots is scored on a different set of plots, so treat the comparison as a rough guide, not a strict test. For a more accurate comparison, use a held-out test split (e.g. 70/30 or 80/20) that stays the same across runs: set the test plots aside once, train each model on the rest only, and score every model on the same test plots. Repeat each run with a few random seeds, because results vary from run to run.

Plot names follow `r<row>_c<col>_<genotype>_<site>_<date>`. If a genotype ID is wrong, it can be corrected later from row/col without changing the masks.

## Current model (104 plots, 5-fold CV)

IoU 0.669 · Dice 0.784 · canopy % MAE 5.74 pts · R² 0.919 · bias +2.55 (slight over-prediction).
U-Net with ResNet34 (ImageNet) encoder, BCE + Dice loss, Adam lr 1e-3, 150 epochs, batch 4, 128 px letterboxed input, threshold 0.5.
Prediction uses 128 px tiles with stride 64 at the orthophoto's native resolution, without resizing. The model was trained at ≈2.3 cm/px, so orthos with a very different GSD should be resampled to ≈2.3 cm/px first.

## Acknowledgements

Developed by Sasi Velchuri with guidance from Boubacar Gano. Training data annotation was contributed by the Shakoor Lab (Donald Danforth Plant Science Center) and the DEFI Team (CIRAD).
