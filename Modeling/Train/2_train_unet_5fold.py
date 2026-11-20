"""
2_train_unet_5fold.py
---------------------
Trains the canopy U-Net with 5-fold CV, then retrains one model on all plots.
These are the exact settings used for the 104-plot model in Modeling/Weight
(its CV results are in Train/results_104 -- the baseline to beat).

To retrain with new annotations: run 1_build_training_data.py, set DATA_DIR to
its OUT_DIR, give OUT_DIR a new name, and run this script. Copy the new
unet_canopy_full.pt into Modeling/Weight only after it beats results_104.

U-Net training with K-fold cross-validation for yam canopy segmentation.
The dataset, model, and loss match train_unet.py, so the results stay comparable.

WHAT K-FOLD CV MEASURES (read this before you read the numbers)  [fix #6]
    Cross-validation estimates how well the training procedure generalises. It
    does not measure the accuracy of one saved model file. Each plot gets one
    out-of-fold prediction from a model that never trained on it, so the
    predictions come from K different models. The script also retrains one model
    on all plots and saves it as unet_canopy_full.pt. The CV estimates that
    model accuracy. It does not measure it directly. Report the mean and the
    standard deviation across plots. Do not rely on any single fold.

FIXES vs the earlier CV version
    #1 ImageNet normalisation. The script normalises inputs with the ImageNet
       mean and standard deviation that the pretrained ResNet34 encoder expects,
       not a bare divide by 255. Training, validation, and prediction must match.
    #2 Loss-curve logging. The script saves the per-epoch training loss for each
       fold and prints a short plateau summary. Use the curves to choose EPOCHS.
       Keep EPOCHS fixed. Early stopping on the validation fold leaks information.
    #4 Letterbox padding. The script keeps each clip aspect ratio and zero-pads it
       to an IMG_SIZE square, instead of squashing it. Padded pixels get alpha=0,
       so the loss and every metric ignore them.

Ground truth = the hand-labelled masks in masks/*_mask.png.
Per-plot metrics: predicted canopy% vs true canopy%, IoU, and Dice.

Outputs (into OUT_DIR):
  val_report_cv.csv               per-plot true vs predicted canopy%, IoU, Dice, fold
  fold_summary.csv                per-fold mean IoU, Dice, and canopy% error
  loss_curves/<tag>.csv           per-epoch training loss for each fold and the full model
  fold_models/unet_fold{k}.pt     the weights from each fold
  unet_canopy_full.pt             model retrained on all plots (for prediction)
  overlays/<plot>_overlay.png     out-of-fold prediction overlay for each plot
  overlays/<plot>_errormap.png    agreement map (green TP, red FP, blue FN)
  RESULTS_INTERPRETATION.txt      plain-language summary of the numbers

Run in the yam-fcover env. The script uses Apple MPS automatically if available.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
TRAIN_DIR = Path(__file__).resolve().parent                 # Modeling/Train
DATA_DIR = TRAIN_DIR / "training_data_104"                   # images/<stem>.tif + masks/<stem>_mask.png
OUT_DIR  = TRAIN_DIR / "train-output" / "unet_cv_104"        # not committed (see .gitignore)

K_FOLDS  = 5            # 5-fold cross-validation
IMG_SIZE = 128          # model input (clips letterboxed to this square)  [#3 later: raise this]
EPOCHS   = 150          # fixed on purpose. use loss_curves/ to justify this number  [#2]
BATCH    = 4
LR       = 1e-3
SEED     = 42

STRATIFY_BY_COVER = True   # spread the cover range across the folds (good for a small dataset)
TRAIN_FULL_MODEL  = True    # after CV, retrain on all plots for downstream prediction
RESUME_FOLDS      = True    # reuse fold_models/unet_fold{k}.pt if it already exists (skip retraining that fold)
USE_LR_SCHEDULER  = False    # cosine LR decay over EPOCHS (smoother late training) - insignificant change. Turning off, needs more data.

# ImageNet statistics the ResNet34 encoder was pretrained with  [#1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", DEVICE)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "overlays").mkdir(exist_ok=True)
(OUT_DIR / "fold_models").mkdir(exist_ok=True)
(OUT_DIR / "loss_curves").mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# letterbox: aspect-preserving resize + zero-pad to square  [#4]
# ----------------------------------------------------------------------
def letterbox(arr, interp):
    """Resize the array so the longest side equals IMG_SIZE, with the aspect ratio
    kept, then centre-pad with zeros to an IMG_SIZE square. This works for
    3-channel images and 2-D masks. Return the canvas and the pad geometry
    (top, left, new_h, new_w) so the caller can invert the padding."""
    H, W = arr.shape[:2]
    scale = IMG_SIZE / max(H, W)
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    resized = np.array(Image.fromarray(arr).resize((new_w, new_h), interp))
    top  = (IMG_SIZE - new_h) // 2
    left = (IMG_SIZE - new_w) // 2
    if resized.ndim == 3:
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, resized.shape[2]), dtype=arr.dtype)
    else:
        canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=arr.dtype)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (top, left, new_h, new_w)

# ----------------------------------------------------------------------
# DATASET   (same content as train_unet.py; letterbox + ImageNet normalize added)
# ----------------------------------------------------------------------
class CanopyDataset(Dataset):
    """Returns letterboxed+normalized RGB, canopy mask, and alpha (in-plot) mask."""
    def __init__(self, stems, augment):
        self.stems = stems
        self.augment = augment
        # geometric and mild photometric augmentation, then ImageNet normalisation.
        # do not shift colour strongly. colour is the canopy signal. no resize here.
        # the letterbox already made an IMG_SIZE square, so the augmentations keep it.  [#1][#4]
        if augment:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                normalize,
            ])
        else:
            self.tf = A.Compose([normalize])

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        rgba, mask = CACHE[stem]                 # preloaded in RAM (no OneDrive reads mid-training)
        rgb = rgba[:, :, :3]
        alpha = (rgba[:, :, 3] > 0).astype(np.uint8)
        canopy = (mask > 128).astype(np.uint8)

        # letterbox: bilinear for the image, nearest for the masks (keeps them binary)  [#4]
        rgb_lb, _ = letterbox(rgb, Image.BILINEAR)
        canopy_lb, _ = letterbox(canopy, Image.NEAREST)
        alpha_lb, _ = letterbox(alpha, Image.NEAREST)

        # apply the same geometric transform to the image and both masks. normalise changes the image only
        out = self.tf(image=rgb_lb, masks=[canopy_lb, alpha_lb])
        img = out["image"]                       # float32 HWC, already ImageNet-normalized [#1]
        canopy_r, alpha_r = out["masks"]

        img = torch.from_numpy(img).permute(2, 0, 1)                 # HWC -> CHW
        canopy_r = torch.from_numpy(canopy_r).float().unsqueeze(0)   # 1HW
        alpha_r = torch.from_numpy(alpha_r).float().unsqueeze(0)
        return img, canopy_r, alpha_r, stem

# ----------------------------------------------------------------------
# loss   (unchanged from train_unet.py: BCE + Dice on in-plot pixels only)
# ----------------------------------------------------------------------
bce = nn.BCEWithLogitsLoss(reduction="none")

def masked_loss(logits, target, alpha):
    """BCE + Dice loss on the in-plot pixels only (alpha>0). Padded corners have
    alpha=0, so the loss ignores them.  [#4]"""
    b = bce(logits, target) * alpha
    bce_loss = b.sum() / (alpha.sum() + 1e-6)
    prob = torch.sigmoid(logits) * alpha
    tgt = target * alpha
    inter = (prob * tgt).sum()
    dice = 1 - (2 * inter + 1e-6) / (prob.sum() + tgt.sum() + 1e-6)
    return bce_loss + dice

# ----------------------------------------------------------------------
# build the cover-stratified folds over all plots
# ----------------------------------------------------------------------
all_stems = sorted(p.stem for p in (DATA_DIR / "images").glob("*.tif"))
print(f"found {len(all_stems)} annotated plots")

# ----------------------------------------------------------------------
# preload every image + mask into RAM once (104 small clips, a few MB).
# Reading from OneDrive every epoch can time out mid-run (Errno 60); after this
# block the training loop never touches the disk for inputs. Each file gets a
# few retries in case OneDrive is still downloading it.
# ----------------------------------------------------------------------
import time
def _load_retry(path, tries=6, wait=10):
    for t in range(1, tries + 1):
        try:
            with Image.open(path) as im:
                return np.array(im)
        except (OSError, TimeoutError) as e:
            if t == tries:
                raise
            print(f"    read failed ({e}); retry {t}/{tries - 1} in {wait}s: {path.name}")
            time.sleep(wait)

def _save_retry(img, path, tries=6, wait=10):
    for t in range(1, tries + 1):
        try:
            img.save(path); return
        except (OSError, TimeoutError) as e:
            if t == tries:
                print(f"    WARNING: could not save {path.name} ({e}); continuing"); return
            time.sleep(wait)

CACHE = {}
for s in all_stems:
    CACHE[s] = (_load_retry(DATA_DIR / "images" / f"{s}.tif"),
                _load_retry(DATA_DIR / "masks" / f"{s}_mask.png"))
print(f"preloaded {len(CACHE)} image/mask pairs into memory")

def true_cover_pct(stem):
    """True (hand-labeled) canopy% for a plot, used only to stratify the folds."""
    rgba, mask = CACHE[stem]
    alpha = rgba[:, :, 3] > 0
    canopy = (mask > 128) & alpha
    return canopy.sum() / alpha.sum() * 100 if alpha.sum() else 0.0

if STRATIFY_BY_COVER:
    ordered = sorted(all_stems, key=true_cover_pct)                 # sparse -> full
    fold_of = {s: (rank % K_FOLDS) for rank, s in enumerate(ordered)}
else:
    rng = np.random.default_rng(SEED)
    perm = list(rng.permutation(all_stems))
    fold_of = {s: (i % K_FOLDS) for i, s in enumerate(perm)}

folds = [[s for s in all_stems if fold_of[s] == k] for k in range(K_FOLDS)]
print("fold sizes:", [len(f) for f in folds])

# ----------------------------------------------------------------------
# train a new model on train_stems. return the model and the per-epoch loss list  [#2]
# ----------------------------------------------------------------------
def train_model(train_stems, tag):
    train_dl = DataLoader(CanopyDataset(train_stems, augment=True),
                          batch_size=BATCH, shuffle=True)
    # same model and optimiser as train_unet.py
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if USE_LR_SCHEDULER else None

    losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot = 0.0
        for img, canopy, alpha, _ in train_dl:
            img, canopy, alpha = img.to(DEVICE), canopy.to(DEVICE), alpha.to(DEVICE)
            opt.zero_grad()
            loss = masked_loss(model(img), canopy, alpha)
            loss.backward(); opt.step()
            tot += loss.item()
        if sched is not None:
            sched.step()                 # cosine LR step, once per epoch
        epoch_loss = tot / len(train_dl)
        losses.append(epoch_loss)
        if epoch % 25 == 0 or epoch == 1:
            print(f"    [{tag}] epoch {epoch:3d}  train_loss {epoch_loss:.4f}")

    # save the loss curve and print a plateau summary, so EPOCHS is evidence-based [#2]
    with open(OUT_DIR / "loss_curves" / f"{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss"])
        w.writerows([(e + 1, round(l, 5)) for e, l in enumerate(losses)])
    best_ep = int(np.argmin(losses)) + 1
    # first epoch where the loss improves by less than 1% over 10 epochs (rough plateau)
    plateau = EPOCHS
    for e in range(10, len(losses)):
        if losses[e - 10] > 0 and (losses[e - 10] - losses[e]) / losses[e - 10] < 0.01:
            plateau = e + 1; break
    print(f"    [{tag}] loss: start {losses[0]:.4f} -> end {losses[-1]:.4f}"
          f"  (min at epoch {best_ep}; ~plateau by epoch {plateau})")
    return model

# ----------------------------------------------------------------------
# validate one plot: hand label vs prediction, at native resolution  [#4 invert the pad]
# ----------------------------------------------------------------------
def validate_plot(model, stem, fold):
    model.eval()
    with torch.no_grad():
        rgba, mask = CACHE[stem]
        H, W = rgba.shape[:2]
        rgb = rgba[:, :, :3]; alpha = rgba[:, :, 3] > 0
        true = (mask > 128)

        # letterbox and normalise exactly like training, and keep the pad geometry [#1][#4]
        canvas, (top, left, nh, nw) = letterbox(rgb, Image.BILINEAR)
        inp = normalize(image=canvas)["image"]
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        prob_sq = torch.sigmoid(model(inp))[0, 0].cpu().numpy()      # IMG_SIZE x IMG_SIZE

        # undo the padding, then resize the real content back to native (W, H)  [#4]
        crop = prob_sq[top:top + nh, left:left + nw]
        prob = np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
        pred = (prob > 0.5) & alpha              # threshold + restrict to in-plot

        true = true & alpha
        inter = (pred & true).sum(); union = (pred | true).sum()
        iou = inter / (union + 1e-6)
        dice = 2 * inter / (pred.sum() + true.sum() + 1e-6)
        pred_pct = pred.sum() / alpha.sum() * 100
        true_pct = true.sum() / alpha.sum() * 100

        ov = rgb.copy(); ov[pred] = [255, 0, 255]; ov[~alpha] = [128, 128, 128]
        _save_retry(Image.fromarray(ov).resize((W * 4, H * 4), Image.NEAREST),
                    OUT_DIR / "overlays" / f"{stem}_overlay.png")

        # error map: where the model agrees or disagrees with the hand label  [#4]
        #   green = true positive (canopy correct)   red = false positive (over-segment)
        #   blue  = false negative (missed canopy)    dim RGB = correct background
        em = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
        tp = pred & true
        fp = pred & (~true) & alpha
        fn = (~pred) & true & alpha
        em[tp] = [0, 220, 0]; em[fp] = [230, 0, 0]; em[fn] = [0, 90, 255]
        em[~alpha] = [128, 128, 128]
        _save_retry(Image.fromarray(em).resize((W * 4, H * 4), Image.NEAREST),
                    OUT_DIR / "overlays" / f"{stem}_errormap.png")

    return (stem, fold, round(float(true_pct), 1), round(float(pred_pct), 1),
            round(float(iou), 3), round(float(dice), 3))

# ----------------------------------------------------------------------
# cross-validation loop
# ----------------------------------------------------------------------
print("\n5-fold cross-validation...")
rows = []                                  # one out-of-fold row per plot
for k in range(K_FOLDS):
    val_stems = folds[k]
    train_stems = [s for s in all_stems if s not in val_stems]
    print(f"\n=== fold {k+1}/{K_FOLDS}   train {len(train_stems)}   val {len(val_stems)} ===")

    fold_pt = OUT_DIR / "fold_models" / f"unet_fold{k+1}.pt"
    if RESUME_FOLDS and fold_pt.exists():
        # resume: this fold finished in an earlier run. Load its weights and only re-validate.
        print(f"    RESUME: loading existing {fold_pt.name} (not retraining this fold)")
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=3, classes=1).to(DEVICE)
        model.load_state_dict(torch.load(fold_pt, map_location=DEVICE))
    else:
        model = train_model(train_stems, tag=f"fold{k+1}")
        torch.save(model.state_dict(), fold_pt)

    for stem in val_stems:
        r = validate_plot(model, stem, k + 1)
        rows.append(r)
        print(f"    {r[0]:32s} true {r[2]:>5} pred {r[3]:>5}  IoU {r[4]:.3f}  Dice {r[5]:.3f}")

# ----------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------
rows.sort(key=lambda r: r[2])              # sort by true canopy% for readability

with open(OUT_DIR / "val_report_cv.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["plot", "fold", "true_canopy_pct", "pred_canopy_pct", "IoU", "Dice"])
    w.writerows(rows)

with open(OUT_DIR / "fold_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "n_val", "mean_IoU", "mean_Dice", "mean_abs_canopy_err_pct"])
    for k in range(1, K_FOLDS + 1):
        fr = [r for r in rows if r[1] == k]
        w.writerow([k, len(fr),
                    round(np.mean([r[4] for r in fr]), 3),
                    round(np.mean([r[5] for r in fr]), 3),
                    round(np.mean([abs(r[3] - r[2]) for r in fr]), 2)])

# aggregate over all out-of-fold plots
iou  = np.array([r[4] for r in rows])
dice = np.array([r[5] for r in rows])
t    = np.array([r[2] for r in rows])
p    = np.array([r[3] for r in rows])
err  = p - t
mae  = float(np.abs(err).mean())
rmse = float(np.sqrt((err ** 2).mean()))
bias = float(err.mean())                                    # positive bias = the model over-predicts cover
r2   = float(1 - ((t - p) ** 2).sum() / (((t - t.mean()) ** 2).sum() + 1e-9))

print(f"\n===== 5-FOLD CV: estimated out-of-fold generalization (all {len(all_stems)} plots) =====")  # [#6]
print(f"{'plot':32s} {'fold':>4} {'true%':>6} {'pred%':>6} {'IoU':>6} {'Dice':>6}")
for r in rows:
    print(f"{r[0]:32s} {r[1]:>4} {r[2]:>6} {r[3]:>6} {r[4]:>6} {r[5]:>6}")
print("-" * 70)
print(f"Segmentation : mean IoU {iou.mean():.3f} +/- {iou.std():.3f}   "
      f"mean Dice {dice.mean():.3f} +/- {dice.std():.3f}")
print(f"Canopy% (Fcover): MAE {mae:.2f}   RMSE {rmse:.2f}   bias {bias:+.2f}   R2 {r2:.3f}")

# plain-language interpretation written to disk  [#6]
with open(OUT_DIR / "RESULTS_INTERPRETATION.txt", "w") as f:
    f.write(
        "5-FOLD CROSS-VALIDATION RESULTS - HOW TO READ THEM\n"
        "==================================================\n\n"
        f"Plots: {len(all_stems)}   Folds: {K_FOLDS}   Epochs/fold: {EPOCHS}   "
        f"Input: {IMG_SIZE}px letterboxed, ImageNet-normalized\n\n"
        "WHAT THESE NUMBERS ARE\n"
        "  These are OUT-OF-FOLD estimates: each plot was scored by a model that never\n"
        f"  saw it in training. The {len(all_stems)} predictions come from {K_FOLDS} different models. The\n"
        "  numbers estimate how well the TRAINING PROCEDURE generalizes -- they are NOT\n"
        "  the measured accuracy of any single saved model.\n\n"
        f"  unet_canopy_full.pt (trained on all {len(all_stems)} plots) is the model you deploy for\n"
        "  prediction. Its true accuracy is ESTIMATED by this CV, not measured directly.\n\n"
        f"HEADLINE (mean over {len(all_stems)} plots; trust mean +/- std, not any one fold at n={len(all_stems)})\n"
        f"  Segmentation   : IoU {iou.mean():.3f} +/- {iou.std():.3f}   "
        f"Dice {dice.mean():.3f} +/- {dice.std():.3f}\n"
        f"  Fcover (canopy%): MAE {mae:.2f} pts   RMSE {rmse:.2f}   "
        f"bias {bias:+.2f}   R2 {r2:.3f}\n"
        f"  (bias > 0 means the model over-predicts canopy cover on average.)\n\n"
        "CAVEATS\n"
        f"  * n={len(all_stems)} is small: fold-to-fold variance is expected and is not model failure.\n"
        "  * Sparse-canopy plots are the hardest. Check them in val_report_cv.csv.\n"
        "  * Treat this as a pipeline sanity check. Re-run after each annotation batch\n"
        "    and watch MAE trend down / R2 up.\n"
        "  * See loss_curves/ to judge whether EPOCHS is sensible before scaling up.\n"
    )

# ----------------------------------------------------------------------
# deployment model: retrain on all plots
# ----------------------------------------------------------------------
if TRAIN_FULL_MODEL:
    print(f"\n=== retraining on all {len(all_stems)} plots for deployment ===")
    full = train_model(all_stems, tag="full")
    torch.save(full.state_dict(), OUT_DIR / "unet_canopy_full.pt")
    print("saved unet_canopy_full.pt  (point predict_unet.py MODEL_PATH here)")

print(f"\noutputs in {OUT_DIR}")
