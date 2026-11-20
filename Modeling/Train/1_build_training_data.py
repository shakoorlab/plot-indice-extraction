"""
1_build_training_data.py
------------------------
Adds new Label Studio annotations to the existing 104-plot training set.

  1. Copies every plot in Train/training_data_104 (MERGE_ORIG) unchanged.
  2. Converts each Label Studio JSON export in JSON_FILES (brush label "canopy")
     into a mask, and copies the matching clip .tif from CLIP_DIRS.
  3. Writes the combined set to OUT_DIR:
       <OUT_DIR>/images/<stem>.tif
       <OUT_DIR>/masks/<stem>_mask.png
       <OUT_DIR>/_overlays/<stem>.png   (check these by eye before training)

training_data_104 itself is never modified.
Stems follow r<row>_c<col>_<id>_<site>_<date>; blank boxes ("_nan_") are skipped.

Needs: label-studio-converter.
Run:   python 1_build_training_data.py
Then set DATA_DIR in 2_train_unet_5fold.py to OUT_DIR and train.
"""

import json, glob, os, re, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from label_studio_converter.brush import decode_rle

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
TRAIN_DIR = Path(__file__).resolve().parent          # Modeling/Train

# EDIT: Label Studio JSON export(s) with the NEW annotations.
JSON_FILES = [
    # "/path/to/new_annotations_export.json",
]
# EDIT: folders searched in order for <stem>.tif (first match wins).
# Point these at the clip folders the new annotations were made from
# (e.g. Yam-fcover/clip_archive/<site>/<date> on the lab OneDrive).
CLIP_DIRS = [
    # "/path/to/clip_archive/<site>/<date>",
]
# existing training set (copied unchanged) and the new combined output
MERGE_ORIG = str(TRAIN_DIR / "training_data_104")
OUT_DIR    = str(TRAIN_DIR / "training_data_new")    # rename as you like, e.g. training_data_<N>
BRUSH_LABEL = "canopy"
# blank boxes (no variety in the field map) are left out of training.
EXCLUDE_STEMS = {"r12_c12_nan_RoujolA_12-12-24"}
EXPECTED_TOTAL = None   # set to 104 + <number of new plots> to get a count check; None = skip

if not JSON_FILES:
    raise SystemExit("Add at least one Label Studio export to JSON_FILES in the CONFIG block.")
if os.path.abspath(OUT_DIR) == os.path.abspath(MERGE_ORIG):
    raise SystemExit("OUT_DIR must differ from MERGE_ORIG (training_data_104 stays unchanged).")

# ----------------------------------------------------------------------
def stem_of(img_path):
    """/data/upload/1/<hash>-r1_c11_165_RoujolA_12-12-24.png -> r1_c11_165_RoujolA_12-12-24"""
    base = os.path.basename(img_path)
    base = re.sub(r'^[0-9a-fA-F]+-', '', base)     # strip LS hash prefix
    return os.path.splitext(base)[0]

def task_mask(task):
    """OR together all canopy brush regions -> (H,W) bool mask at LS resolution."""
    regions = [r for a in task.get('annotations', []) for r in a.get('result', [])
               if r.get('type') == 'brushlabels' and BRUSH_LABEL in r['value'].get('brushlabels', [])]
    if not regions:
        return None
    H = regions[0]['original_height']; W = regions[0]['original_width']
    m = np.zeros((H, W), bool)
    for r in regions:
        out = decode_rle(r['value']['rle'])
        m |= (np.reshape(out, [r['original_height'], r['original_width'], 4])[:, :, 3] > 128)
    return m

def find_clip(stem):
    """Search CLIP_DIRS in order for <stem>.tif. Returns the path, or None."""
    for d in CLIP_DIRS:
        cand = f"{d}/{stem}.tif"
        if os.path.exists(cand):
            return cand
    return None

out = Path(OUT_DIR); (out / "masks").mkdir(parents=True, exist_ok=True)
if CLIP_DIRS: (out / "images").mkdir(exist_ok=True); (out / "_overlays").mkdir(exist_ok=True)

seen, conflicts, rows, missing_tif, failed = {}, [], [], [], []

def image_mask_ready(s):
    """True only if BOTH the image and mask already exist and are non-empty
    (lets a re-run skip completed plots and retry ones an earlier run failed on)."""
    img = out / "images" / f"{s}.tif"
    mk  = out / "masks" / f"{s}_mask.png"
    return mk.exists() and mk.stat().st_size > 0 and (
        not CLIP_DIRS or (img.exists() and img.stat().st_size > 0)
    )

# ---- fold in the original training_data (optional) ----
if MERGE_ORIG:
    for tif in sorted(glob.glob(f"{MERGE_ORIG}/images/*.tif")):
        s = Path(tif).stem
        if s in EXCLUDE_STEMS or "_nan_" in s:
            continue                                   # skip blank boxes
        mk = f"{MERGE_ORIG}/masks/{s}_mask.png"
        if os.path.exists(mk):
            if image_mask_ready(s):
                seen[s] = "original"; continue
            try:
                shutil.copy(tif, out / "images" / f"{s}.tif") if CLIP_DIRS else None
                shutil.copy(mk, out / "masks" / f"{s}_mask.png")
                seen[s] = "original"
            except OSError as e:
                failed.append((s, "original", str(e)))

# ---- convert each LS export ----
for jf in JSON_FILES:
    proj = os.path.basename(jf)
    tasks = json.load(open(jf))
    n = 0
    for t in tasks:
        s = stem_of(t['data']['image'])
        if s in EXCLUDE_STEMS or "_nan_" in s:
            continue                                   # skip blank boxes
        if s in seen:
            conflicts.append((s, seen[s], proj)); continue      # keep first, log conflict
        if image_mask_ready(s):
            seen[s] = proj; n += 1; continue                    # already done (resumed run)
        m = task_mask(t)
        if m is None:
            continue

        # resize mask to the .tif's native size if the clip is available.
        # Everything that touches disk for this plot is wrapped in one try/except:
        # a bare OSError here (OneDrive write hiccup, transient lock) used to propagate
        # and silently truncate the whole run after this plot. Now it is caught, the
        # plot is logged as failed (and retried automatically next run), and the loop
        # continues instead of aborting.
        try:
            tif = find_clip(s)
            if tif:
                rgba = np.array(Image.open(tif)); H, W = rgba.shape[:2]
                mask = np.array(Image.fromarray((m*255).astype(np.uint8)).resize((W, H), Image.NEAREST))
                shutil.copy(tif, out / "images" / f"{s}.tif")
                # sanity overlay: magenta canopy on the clip, gray outside plot
                rgb = rgba[:, :, :3].copy(); alpha = rgba[:, :, 3] > 0
                ov = rgb.copy(); ov[(mask > 128) & alpha] = [255, 0, 255]; ov[~alpha] = [128, 128, 128]
                Image.fromarray(ov).resize((W*4, H*4), Image.NEAREST).save(out / "_overlays" / f"{s}.png")
                frac = ((mask > 128) & alpha).sum() / alpha.sum() * 100
            else:
                if CLIP_DIRS: missing_tif.append(s)
                mask = (m*255).astype(np.uint8); frac = (m).mean()*100
            Image.fromarray(mask).save(out / "masks" / f"{s}_mask.png")
        except OSError as e:
            failed.append((s, proj, str(e)))
            print(f"  WARNING: {s} failed ({e}) -- will retry on next run")
            continue

        seen[s] = proj
        rows.append((s, proj, round(float(frac), 1))); n += 1
    print(f"{proj}: converted {n} masks")

# ---- report ----
print(f"\nTOTAL plots in {OUT_DIR}: {len(seen)}")
if MERGE_ORIG: print(f"  (includes {sum(v=='original' for v in seen.values())} original)")
print("duplicate/conflicting stems (kept first):", conflicts if conflicts else "none")
if CLIP_DIRS: print("labeled tasks with NO matching .tif:", len(missing_tif), missing_tif[:6])
if failed:
    print(f"\n*** {len(failed)} plot(s) FAILED to write (transient I/O) -- run this script again to retry: ***")
    for s, proj, err in failed:
        print(f"  {s}  (from {proj}): {err}")
else:
    print("no I/O failures")
rows.sort(key=lambda r: r[2])
print("\ncanopy% by plot (sparse -> dense):")
for s, p, f in rows:
    print(f"  {s:32s} {f:5.1f}%")
print(f"\noutputs in {out.resolve()}")
if EXPECTED_TOTAL is None:
    print(f"{len(seen)} plots (no EXPECTED_TOTAL set).")
elif len(seen) != EXPECTED_TOTAL:
    print(f"\n*** WARNING: expected {EXPECTED_TOTAL} plots, found {len(seen)}. Check before training. ***")
else:
    print(f"OK: {len(seen)} plots, as expected.")
