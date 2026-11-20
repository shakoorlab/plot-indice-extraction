"""
full_predict_ortho.py   (Full_Canopy_Cover -- step 1)
-----------------------------------------------------
Same method as Test/predict_ortho_104.py, for running every site/date in the master
workbook (USE_XLSX_DATES). Same method. Writes per site/date:
  Canopy_Probability/<site>/<site>_<date>_canopy_prob.tif   (read by full_predict_watershed.py)
  Estimates/<site>/<date>/<site>_<date>_predict.png          (canopy magenta, plot boxes red)

predict_ortho.py   (PredictV2 -- step 1)
----------------------------------------
Run the CURRENT U-Net over the WHOLE orthophoto for each site/date, then read
each plot's canopy area from that one field-wide canopy map.

Purpose: a quick check that the model behaves the same on the full field as it
does on the single black-surround clips. If the per-plot areas here match
predict_unet.py's areas, the field-wide map can be trusted as the input to the
watershed step (predict_watershed.py, next).

How plot identity works without clips
  The GRID shapefile for each site/date carries row, col and var (genotype)
  for every polygon -- the same source current-reclip_batch.py uses to name
  the clips. Here the polygons are laid over the field canopy map AFTER
  prediction instead of being used to cut clips BEFORE prediction. The plot
  name ("file" column) is built exactly like the clip name, so rows join 1:1
  with the old predictions CSV.

Pixel-for-pixel consistency with the clips
  Plot pixels are rasterised with all_touched=False, which is the same rule
  rasterio.mask (used to cut the clips) uses. So in_plot_px should match the
  old CSV exactly. canopy_px will differ a little because the model now sees
  the real surroundings instead of black padding -- that difference is what
  this step measures.

Outputs per site/date, in PredictV2/Predictions/<site>/:
  <site>_<date>_ortho.csv           same columns as predict_unet.py, minus fcover_%
  <site>_<date>_ortho_compare.csv   old vs new canopy area per plot (if old CSV found)
  <site>_<date>_canopy_prob.tif     field canopy probability (0-255), georeferenced,
                                    reused by the watershed step so the model
                                    doesn't have to run twice
  <site>_<date>_field_overlay.png   whole field: canopy magenta, plot outlines red
  overlays_ortho/<plot>_ortho.png   per plot (+ margin): canopy inside the box magenta,
                                    canopy OUTSIDE the box cyan (= spill-over the
                                    watershed will have to deal with), box outline red

Run in the yam-fcover env (needs torch, segmentation_models_pytorch, rasterio,
geopandas, numpy, Pillow).
"""

import csv
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import from_bounds

# ----------------------------------------------------------------------
# CONFIG -- edit these
# ----------------------------------------------------------------------
MODELING = Path(__file__).resolve().parents[1]      # Modeling/ folder in the repo

# EDIT if needed: the lab "Yam/DroneData/Orthophotos" folder (orthos + GRID shapefiles under Photos/).
# Default = the Shakoor Lab OneDrive path on a Mac, for whoever is logged in.
ROOT = Path.home() / "Library/CloudStorage/OneDrive-DDPSC/Shakoor Lab (Notes)/Yam/DroneData/Orthophotos"
PHOTOS = ROOT / "Photos"

MODEL_PATH = MODELING / "Weight/unet_canopy_full.pt"   # 104-plot model (Train/2_train_unet_5fold.py)

# One entry per site, each with its own list of dates (YYYY-MM-DD folder names under RGB/).
SITE_DATES = {
    "RoujolA": ["2024-11-20"],
    "RoujolD": ["2024-11-20"],
    "GodetC1": ["2024-11-29"],
    "Godet25-26": ["2025-11-12"],
}

# Dates from the master workbook. True = SITE_DATES above is replaced by every
# site/date in the first sheet ("Combined"); False = SITE_DATES used as is.
DATES_XLSX = ROOT / "Yam-fcover/1RGB_MasterDataset_Split_By_Site.xlsx"   # Site/Date list (lab OneDrive)
USE_XLSX_DATES = True
XLSX_TO_SCRIPT_SITE = {"Godet25": "Godet25-26"}   # workbook site name -> name used here

# Where each site's RGB/<date>/ folders live (same as current-reclip_batch.py).
SITE_FOLDERS = {
    "RoujolA": [PHOTOS / "2024-25" / "RoujolA"],
    "RoujolD": [PHOTOS / "2024-25" / "RoujolD"],
    "GodetC1": [PHOTOS / "2024-25" / "GodetC1", PHOTOS / "2024-25_1" / "GodetC1_2"],
    "Godet25-26": [PHOTOS / "2025-26_part1_extracted" / "godet_2025"],
}

# Sites whose shapefile only has an 'id' column (no row/col). Row/col is rebuilt from the
# field map with the same, already-verified function the clip archive was built with.
ID_ONLY_SITES = {"Godet25-26"}
RECLIP_SCRIPT = MODELING / "Clipping/current-reclip_batch.py"

# ---- NEGATE: site/dates to skip entirely (no processing, no outputs) ----
#   ("Site", "YYYY-MM-DD") skips one date;  ("Site", "*") skips every date of that site
NEGATE = [
    # ("GodetC1", "2024-11-29"),
    # ("Godet25-26", "*"),
]

FCC_ROOT = MODELING / "Full_canopy_Estimate"          # outputs not committed (see .gitignore)
OUT_ROOT = FCC_ROOT / "Canopy_Probability"          # canopy_prob.tif per site/date
ESTIMATES_ROOT = FCC_ROOT / "Estimates"             # <site>/<date>/<site>_<date>_predict.png

SKIP_EXISTING = False    # True = skip a site/date whose canopy_prob.tif already exists (resume)

# Old per-clip predictions to compare against (predict_unet.py output layout).
OLD_PRED_ROOT = ROOT / "Yam-fcover/Modeling/Test/predict_results/Predictions"
COMPARE_TO_OLD = False

TILE = 128              # model input size; tiles are cut at native resolution (no resizing)
STRIDE = 64             # tile step; 64 = 50% overlap, every pixel predicted up to 4 times and averaged
BATCH = 32              # tiles per forward pass
THRESHOLD = 0.5         # same threshold as predict_unet.py
FIELD_MARGIN_M = 3.0    # metres of orthophoto kept around the plots (context for edge plots)

SAVE_FIELD_OVERLAY = True
SAVE_PLOT_OVERLAYS = False

# Output PNGs for these sites are rotated 90 degrees counter-clockwise (display only; areas unaffected).
ROTATE_CCW_SITES = {"GodetC1"}
PLOT_OVERLAY_MARGIN_M = 0.75   # metres shown around each box in the per-plot overlay

F_ROW, F_COL, F_GENO = "row", "col", "var"   # shapefile attribute names

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CSV_HEADER = ["file", "row", "col", "date", "genotype", "in_plot_px", "canopy_px",
              "canopy_area_cm2", "segmentation_area_cm2",
              "canopy_area_m2", "segmentation_area_m2"]


# ----------------------------------------------------------------------
# finding inputs (same rules as current-reclip_batch.py)
# ----------------------------------------------------------------------
def resolve_site_folder(site, date_str):
    for folder in SITE_FOLDERS.get(site, []):
        if (folder / "RGB" / date_str).is_dir():
            return folder
    return None


def pick_orthomosaic(date_dir):
    all_tifs = list(date_dir.glob("*.tif"))
    if not all_tifs:
        return None, "NO orthomosaic tif found"
    non_ms = [p for p in all_tifs if "mns" not in p.stem.lower() and "dsm" not in p.stem.lower()]
    cands = non_ms if non_ms else all_tifs
    if len(cands) > 1:
        return None, f"AMBIGUOUS orthomosaic: {[c.name for c in cands]}"
    return cands[0], None


def pick_grid_shapefile(date_dir):
    cands = [p for p in date_dir.rglob("*.shp") if "bad" not in str(p).lower()]
    if len(cands) != 1:
        return None, f"{len(cands)} grid shapefile candidates: {[c.name for c in cands]}"
    return cands[0], None


def is_negated(site, date_str):
    return (site, date_str) in NEGATE or (site, "*") in NEGATE


def load_site_dates():
    """{site: sorted unique YYYY-MM-DD dates} from the workbook's first sheet, or SITE_DATES."""
    if not USE_XLSX_DATES:
        return SITE_DATES
    df = pd.read_excel(DATES_XLSX, sheet_name=0, usecols=["Site", "Date"]).dropna()
    df["Site"] = df["Site"].astype(str).str.strip().replace(XLSX_TO_SCRIPT_SITE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return {s: sorted(g["Date"].unique()) for s, g in df.groupby("Site", sort=False)}


def find_inputs(site, date_str):
    """Return (orthophoto, shapefile, problem), found under <site folder>/RGB/<date>/."""
    site_folder = resolve_site_folder(site, date_str)
    if site_folder is None:
        return None, None, f"no RGB/{date_str} folder found"
    date_dir = site_folder / "RGB" / date_str
    o, g = pick_orthomosaic(date_dir), pick_grid_shapefile(date_dir)
    o = o[0] if isinstance(o, tuple) else o
    g = g[0] if isinstance(g, tuple) else g
    if o is None or g is None:
        return None, None, "orthophoto or GRID shapefile not found / ambiguous"
    return o, g, None


_RECLIP = None


def id_only_rowcol(gdf):
    """Row/col for an id-only shapefile (Godet25), via current-reclip_batch.py's field-map logic."""
    global _RECLIP
    if _RECLIP is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("reclip_batch", RECLIP_SCRIPT)
        _RECLIP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RECLIP)
    row_map = _RECLIP.load_godet25_map(_RECLIP.GODET25_MAP)
    return _RECLIP.assign_godet25_row_col(gdf, row_map)


def site_identities(gdf, site, date_tag):
    """One (name, row, col, genotype) per gdf row (same order), or None if unusable."""
    if site not in ID_ONLY_SITES:
        return [plot_identity(r, site, date_tag) for _, r in gdf.iterrows()]
    assignment, m, t, note = id_only_rowcol(gdf)
    if assignment is None:
        raise ValueError(f"row/col reconstruction from the field map failed: {note}")
    print(f"  row/col rebuilt from field map: {m}/{t} id agreement")
    out = []
    for rc, (_, r) in zip(assignment, gdf.iterrows()):
        if rc is None:
            out.append(None)
            continue
        raw = str(r["id"]).strip()
        if raw.lower() in ("nan", "", "none"):
            geno = "nan"
        else:
            try:
                geno = str(int(float(raw)))
            except ValueError:
                geno = raw
        out.append((f"r{rc[0]}_c{rc[1]}_{geno}_{site}_{date_tag}", rc[0], rc[1], geno))
    return out


def plot_identity(r, site, date_tag):
    """Return (name, row, col, geno) built exactly like the clip filenames, or None."""
    try:
        row_v, col_v = int(r[F_ROW]), int(r[F_COL])
    except Exception:
        return None
    raw = str(r[F_GENO]).strip()
    if raw.lower() in ("nan", "", "none"):
        geno = "nan"   # clip archive keeps these as 'nan' plots
    else:
        try:
            geno = str(int(float(raw)))
        except ValueError:
            geno = raw
    return f"r{row_v}_c{col_v}_{geno}_{site}_{date_tag}", row_v, col_v, geno


# ----------------------------------------------------------------------
# model + tiled inference
# ----------------------------------------------------------------------
def load_model():
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()


def tile_starts(n):
    if n <= TILE:
        return [0]
    starts = list(range(0, n - TILE + 1, STRIDE))
    if starts[-1] != n - TILE:
        starts.append(n - TILE)
    return starts


def predict_field(model, rgb):
    """rgb: HxWx3 uint8 (nodata already set to 0). Returns HxW float32 canopy probability."""
    H, W = rgb.shape[:2]
    pad_h, pad_w = max(0, TILE - H), max(0, TILE - W)
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = rgb.shape[:2]

    norm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD   # same as A.Normalize
    prob_sum = np.zeros((Hp, Wp), np.float32)
    count = np.zeros((Hp, Wp), np.float32)

    coords = [(y, x) for y in tile_starts(Hp) for x in tile_starts(Wp)]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(coords), BATCH):
            batch = coords[i:i + BATCH]
            arr = np.stack([norm[y:y + TILE, x:x + TILE] for y, x in batch])        # B,T,T,3
            inp = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
            out = torch.sigmoid(model(inp))[:, 0].cpu().numpy()                    # B,T,T
            for (y, x), p in zip(batch, out):
                prob_sum[y:y + TILE, x:x + TILE] += p
                count[y:y + TILE, x:x + TILE] += 1
    print(f"  predicted {len(coords)} tiles in {time.time() - t0:.0f}s")
    return (prob_sum / np.maximum(count, 1))[:H, :W]


# ----------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------
def outline(mask):
    """Boundary pixels of a boolean mask (4-neighbour erosion, numpy only)."""
    e = mask.copy()
    e[1:] &= mask[:-1]; e[:-1] &= mask[1:]
    e[:, 1:] &= mask[:, :-1]; e[:, :-1] &= mask[:, 1:]
    return mask & ~e


def save_field_overlay(rgb, canopy, plot_labels, path, max_side=3000, rotate_ccw=False):
    ov = rgb.copy()
    ov[canopy] = (0.45 * ov[canopy] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    ov[outline(plot_labels > 0)] = [255, 0, 0]
    img = Image.fromarray(ov)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.NEAREST)
    if rotate_ccw:
        img = img.transpose(Image.Transpose.ROTATE_90)
    img.save(path)


def save_plot_overlay(rgb, canopy, in_plot, margin_px, path):
    rr, cc = np.nonzero(in_plot)
    r0, r1 = max(0, rr.min() - margin_px), min(rgb.shape[0], rr.max() + margin_px + 1)
    c0, c1 = max(0, cc.min() - margin_px), min(rgb.shape[1], cc.max() + margin_px + 1)
    sub_rgb = rgb[r0:r1, c0:c1].copy()
    sub_can = canopy[r0:r1, c0:c1]
    sub_in = in_plot[r0:r1, c0:c1]
    sub_rgb[sub_can & sub_in] = [255, 0, 255]     # counted (inside box)
    sub_rgb[sub_can & ~sub_in] = [0, 200, 255]    # canopy outside the box
    sub_rgb[outline(sub_in)] = [255, 0, 0]
    h, w = sub_rgb.shape[:2]
    Image.fromarray(sub_rgb).resize((w * 4, h * 4), Image.NEAREST).save(path)


def compare_with_old(site, date_str, date_tag, new_rows, out_dir):
    old_csv = OLD_PRED_ROOT / site / f"{site}_{date_str}.csv"
    if not old_csv.exists():
        print(f"  (no old CSV at {old_csv} -- skipping comparison)")
        return
    old = {}
    with open(old_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("file") or r.get("plot")
            old[key] = r
    pairs = []
    for r in new_rows:
        o = old.get(r[0])
        if o is None:
            continue
        old_m2, new_m2 = float(o["canopy_area_m2"]), float(r[9])
        old_px, new_px = int(o["in_plot_px"]), int(r[5])
        pairs.append((r[0], old_px, new_px, old_m2, new_m2, new_m2 - old_m2,
                      (new_m2 - old_m2) / old_m2 * 100 if old_m2 > 0 else float("nan")))
    if not pairs:
        print("  (no matching plot names between old and new CSV)")
        return
    with open(out_dir / f"{site}_{date_str}_ortho_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "old_in_plot_px", "new_in_plot_px", "old_canopy_m2", "new_canopy_m2",
                    "diff_m2", "diff_pct"])
        w.writerows([(p[0], p[1], p[2], round(p[3], 4), round(p[4], 4), round(p[5], 4),
                      round(p[6], 1)) for p in pairs])
    a = np.array([[p[3], p[4]] for p in pairs])
    d = a[:, 1] - a[:, 0]
    px_match = sum(p[1] == p[2] for p in pairs)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1] if len(a) > 2 else float("nan")
    print(f"  COMPARE vs predict_unet.py ({len(pairs)} plots matched by name):")
    print(f"    plot pixels identical : {px_match}/{len(pairs)}  (should be all -- same polygons, same rule)")
    print(f"    canopy m2  old mean {a[:, 0].mean():.3f}   new mean {a[:, 1].mean():.3f}")
    print(f"    bias (new-old)        : {d.mean():+.3f} m2   mean |diff| {np.abs(d).mean():.3f} m2")
    print(f"    correlation r         : {r:.3f}")


# ----------------------------------------------------------------------
def run_site_date(model, site, date_str):
    date_tag = f"{date_str[5:7]}-{date_str[8:10]}-{date_str[2:4]}"   # YYYY-MM-DD -> MM-DD-YY
    ortho, grid, problem = find_inputs(site, date_str)
    if problem:
        print(f"[{site} {date_str}] SKIP -- {problem}")
        return None
    print(f"\n[{site} {date_str}] ortho={ortho.name}  grid={grid.name}")

    gdf = gpd.read_file(grid)
    missing = [c for c in (F_ROW, F_COL, F_GENO) if c not in gdf.columns]
    if site in ID_ONLY_SITES:
        missing = [] if "id" in gdf.columns else ["id"]
    if missing:
        print(f"  SKIP -- shapefile missing columns {missing}")
        return None

    with rasterio.open(ortho) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        minx, miny, maxx, maxy = gdf.total_bounds
        m = FIELD_MARGIN_M
        win = from_bounds(minx - m, miny - m, maxx + m, maxy + m, src.transform).round_offsets().round_lengths()
        data = src.read(window=win, boundless=True, fill_value=0)
        tf = src.window_transform(win)
        crs = src.crs
    gsd_x, gsd_y = abs(tf.a), abs(tf.e)
    px_area_cm2 = (gsd_x * 100) * (gsd_y * 100)

    rgb = np.transpose(data[:3], (1, 2, 0)).copy()
    if data.shape[0] >= 4:
        rgb[data[3] == 0] = 0                     # nodata -> black, as in the clips
    H, W = rgb.shape[:2]
    print(f"  field window {W}x{H} px, GSD {gsd_x * 100:.2f} cm, {len(gdf)} polygons")

    # plot label image (same pixel rule as rasterio.mask used for the clips)
    idents = []
    shapes = []
    all_idents = site_identities(gdf, site, date_tag)
    for i, (_, r) in enumerate(gdf.iterrows()):
        ident = all_idents[i]
        if ident is None or r.geometry is None or r.geometry.is_empty:
            continue
        idents.append(ident)
        shapes.append((r.geometry, len(idents)))
    labels = rasterize(shapes, out_shape=(H, W), transform=tf, fill=0, all_touched=False, dtype="int32")

    prob = predict_field(model, rgb)
    canopy = prob > THRESHOLD

    out_dir = OUT_ROOT / site
    out_dir.mkdir(parents=True, exist_ok=True)

    # probability map for the watershed step
    meta = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs,
                transform=tf, compress="deflate")
    with rasterio.open(out_dir / f"{site}_{date_str}_canopy_prob.tif", "w", **meta) as dst:
        dst.write((prob * 255).round().astype(np.uint8), 1)

    # per-plot areas
    n_lab = len(idents) + 1
    in_plot_px = np.bincount(labels.ravel(), minlength=n_lab)
    canopy_px = np.bincount(labels.ravel(), weights=canopy.ravel(), minlength=n_lab).astype(int)

    rows = []
    for k, (name, row_v, col_v, geno) in enumerate(idents, start=1):
        ip, cp = int(in_plot_px[k]), int(canopy_px[k])
        if ip == 0:
            print(f"  WARNING: {name} has 0 pixels in the orthophoto window -- skipped")
            continue
        area_cm2, seg_cm2 = cp * px_area_cm2, ip * px_area_cm2
        rows.append((name, row_v, col_v, date_str, geno, ip, cp,
                     round(area_cm2, 1), round(seg_cm2, 1),
                     round(area_cm2 / 1e4, 4), round(seg_cm2 / 1e4, 4)))
    rows.sort(key=lambda r: (r[1], r[2]))

    print(f"  wrote {site}_{date_str}_canopy_prob.tif ({len(rows)} plots)")

    if SAVE_FIELD_OVERLAY:
        est_dir = ESTIMATES_ROOT / site / date_str
        est_dir.mkdir(parents=True, exist_ok=True)
        save_field_overlay(rgb, canopy, labels, est_dir / f"{site}_{date_str}_predict.png",
                           rotate_ccw=site in ROTATE_CCW_SITES)
    if SAVE_PLOT_OVERLAYS:
        ov_dir = out_dir / "overlays_ortho"
        ov_dir.mkdir(exist_ok=True)
        margin_px = int(round(PLOT_OVERLAY_MARGIN_M / gsd_x))
        for k, (name, *_rest) in enumerate(idents, start=1):
            in_plot = labels == k
            if in_plot.any():
                save_plot_overlay(rgb, canopy, in_plot, margin_px, ov_dir / f"{name}_ortho.png")

    if COMPARE_TO_OLD:
        compare_with_old(site, date_str, date_tag, rows, out_dir)
    return len(rows)


def main():
    print("device:", DEVICE)
    model = load_model()
    summary = []
    for site, dates in load_site_dates().items():
        for d in dates:
            if is_negated(site, d):
                print(f"[{site} {d}] NEGATED -- skipped (see NEGATE in CONFIG)")
                continue
            if SKIP_EXISTING and (OUT_ROOT / site / f"{site}_{d}_canopy_prob.tif").exists():
                print(f"[{site} {d}] SKIP (exists)")
                continue
            try:
                n = run_site_date(model, site, d)
            except Exception as e:
                print(f"[{site} {d}] FAILED: {e}")
                n = None
            summary.append((site, d, n))
    print("\n===== SUMMARY =====")
    for site, d, n in summary:
        status = f"{n} plots" if n is not None else "skipped / failed"
        print(f"  {site:8s} {d}   {status}   -> {OUT_ROOT / site}")


if __name__ == "__main__":
    main()
