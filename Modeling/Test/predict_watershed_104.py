"""
predict_watershed_104.py   (step 2 of 2; run predict_ortho_104.py first)
Edit ROOT in the CONFIG block to point at the lab orthophoto folder (same as step 1).

predict_watershed.py   (PredictV2 / Watershed -- step 2)
--------------------------------------------------------
Assign canopy that grows BEYOND each plot box to the right plot, and split
canopy where neighbouring plants touch, using a seeded watershed.

Input (nothing is re-predicted -- the model does not run here):
  * Modeling/Test/Predictions/<site>/<site>_<date>_canopy_prob.tif   (from predict_ortho.py)
  * the same RGB orthophoto + GRID shapefile predict_ortho.py used

How it works ("taps and water"):
  1. canopy  = model probability > THRESHOLD, and NOT orthophoto no-data
               (no-data mask: the blank image edges are never canopy)
  2. seeds   = the canopy inside each plot's own box. Everything inside your
               own box is always yours -- same as today -- so the watershed
               area can never be smaller than the box area.
  3. flood   = water spreads from every seed at the same time, only across
               canopy pixels. Each canopy pixel goes to the plot whose water
               reaches it first (shortest path THROUGH canopy). Where two
               plants touch, the boundary forms where their water meets.
  4. never reached = canopy not connected to any plot (isolated weeds, field
               border grass not touching a plot) -> not counted.
  5. territory cap (USE_TERRITORY_CAP, on/off switch) = a plot can't claim
               canopy farther than REACH from its own box. REACH is the median
               gap between neighbouring boxes on that date, measured
               automatically from the polygons -- no hand-set number. Between
               neighbours the flood already decides the boundary; the cap only
               stops runaways (edge plots flowing into border grass, a plot
               running down a weedy furrow).

FIX LAYERS (added one at a time, each with its own on/off switch in CONFIG):
  layer 1  SKIP_NAN_PLOTS   blank-genotype ('nan') boxes are not seeds and not reported.
                            Their boxes are drawn in grey on the overlay so you can see them.
  layer 2  NAN_COMPETE      nan boxes still flood like any plot (claiming the grass/weeds
                            around them, so neighbours can't take it), but their result is
                            thrown away and never reported (needs SKIP_NAN_PLOTS = True).
  layer 3  FIELD_BOUNDARY   one outline around the whole plot grid (+ half a gap). Canopy
                            outside it (border grass, drone pad, field edge) can't be
                            claimed or flooded through. Drawn as a light line on the overlay.
                            Currently OFF: the outline was too tight on staggered edge plots
                            and cut off their real canopy.
  layer 4  CORE_SEEDS       the flood starts from the CENTRE of each box (box shrunk by
           + DISTANCE_LANDSCAPE  CORE_SHRINK of its short side) instead of the whole box, and
                            fills thick canopy before thin links, so where two plants touch
                            the boundary lands at the narrow point between them. A neighbour
                            can now win canopy near your box edge if it is really its plant;
                            box canopy nobody else claims still stays yours. Lost box canopy
                            is reported in box_canopy_lost_px.
  layer 5  DIRECTIONAL_REACH the territory cap gets two limits instead of one: REACH along
                            the row (end to end) and REACH across the row (side to side),
                            each = that date's median gap in that direction, measured
                            automatically from the boxes. Plants spread further along the
                            row than across it, so a single limit was cutting real canopy
                            off in a straight line (e.g. RoujolD r1_c1).

Output files keep the SAME names on every run and are overwritten, so the
Predictions folder always shows the latest version. The very first pass is
kept separately in Watershed/FirstPass/. Which layers were on for the current
files is written to <site>_<date>_run_settings.txt.

Does NOT change predict_unet.py, predict_ortho.py or any existing output.
Everything is written under Modeling/Test/Watershed/Predictions/.

Outputs per site/date, in Modeling/Test/Watershed/Predictions/<site>/:
  <site>_<date>_watershed.csv
      same columns as predict_ortho.py (box values) PLUS:
      watershed_px, watershed_area_cm2, watershed_area_m2  canopy assigned by the watershed
      gain_m2          watershed - box  (how much grew beyond the box)
      cut_by_cap_px    canopy this plot reached but lost to the territory cap
                       (> 0 = may still be an undercount, worth a look)
      neighbours       plots whose watershed region touches this one
  <site>_<date>_watershed_overlay.png
      whole field. Each plot's canopy in its own colour (5 colours, never the
      same as a neighbour left/right/above/below/diagonal), canopy NOT counted
      in dark grey, box outlines white.
  overlays/<plot>.png              per-plot crop of the same overlay
  <site>_<date>_run_settings.txt   which switches / layers produced these files

Requires scikit-image + scipy (pip install scikit-image if missing).
Run in the yam-fcover env.
"""

import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import from_bounds

try:
    from scipy import ndimage
    from skimage.segmentation import watershed
except ImportError as e:
    raise SystemExit(f"missing package ({e}). In the yam-fcover env run:  pip install scikit-image scipy")

# ----------------------------------------------------------------------
# CONFIG -- edit these
# ----------------------------------------------------------------------
MODELING = Path(__file__).resolve().parents[1]      # Modeling/ folder in the repo

# EDIT if needed: the lab "Yam/DroneData/Orthophotos" folder (orthos + GRID shapefiles under Photos/).
# Default = the Shakoor Lab OneDrive path on a Mac, for whoever is logged in.
ROOT = Path.home() / "Library/CloudStorage/OneDrive-DDPSC/Shakoor Lab (Notes)/Yam/DroneData/Orthophotos"
PHOTOS = ROOT / "Photos"

SITE_DATES = {
    "RoujolA": ["2024-11-20"],
    "RoujolD": ["2024-11-20"],
    "GodetC1": ["2024-11-29"],
    "Godet25-26": ["2025-11-12"],
}

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

PREDICTV2 = MODELING / "Test"                          # outputs not committed (see .gitignore)
PROB_ROOT = PREDICTV2 / "Predictions"                   # where predict_ortho.py saved canopy_prob.tif
OUT_ROOT = PREDICTV2 / "Watershed" / "Predictions"

# ---- HANDOFF: simple results for sharing, kept apart from the development outputs ----
# One flat folder: <site>_<date>.csv (plot identity + watershed canopy area only)
# + <site>_<date>.png (whole-field watershed overlay).
WRITE_HANDOFF = True
HANDOFF_DIR = PREDICTV2 / "Watershed" / "preliminary_predict_resultsV2"
HANDOFF_HEADER = ["file", "row", "col", "date", "genotype", "canopy_area_cm2", "canopy_area_m2"]

THRESHOLD = 0.5            # same as predict_ortho.py / predict_unet.py

USE_TERRITORY_CAP = True   # True = stop runaways at REACH from own box. False = pure watershed.
REACH_FACTOR = 1.0         # REACH = this x the date's median gap between neighbouring boxes

# ---- fix layers (see docstring) ----
SKIP_NAN_PLOTS = True      # layer 1: True = ignore blank-genotype ('nan') boxes entirely
NAN_COMPETE = True         # layer 2: True = nan boxes compete for canopy but are not reported
FIELD_BOUNDARY = False     # layer 3: OFF -- outline was cutting off real canopy on edge plots
FIELD_MARGIN_FACTOR = 0.5  #          outline = grid hull + this x median gap
CORE_SEEDS = True          # layer 4: True = flood starts from each box's centre, not the whole box
CORE_SHRINK = 0.2          #          box shrunk by this fraction of its short side on every edge
DISTANCE_LANDSCAPE = True  # layer 4: True = fill thick canopy first, split plants at narrow points
DIRECTIONAL_REACH = True   # layer 5: True = separate cap along vs across the row (each = that
                           #          direction's median gap x REACH_FACTOR)

SAVE_FIELD_OVERLAY = True
SAVE_PLOT_OVERLAYS = True
OVERLAY_MAX_SIDE = 5000    # field overlay is downscaled only if bigger than this

F_ROW, F_COL, F_GENO = "row", "col", "var"

# 5 main colours (assigned so neighbours never share one) + spares used only
# if the grid is irregular somewhere. No greens (would blend with canopy) and
# no white (box outlines) or dark grey (canopy not counted).
PALETTE = np.array([
    [255,   0, 255],   # magenta
    [  0, 220, 255],   # cyan
    [255, 235,   0],   # yellow
    [255, 120,   0],   # orange
    [ 60,  90, 255],   # blue
    [170,  70, 255],   # violet   (spare)
    [255,  90, 150],   # pink     (spare)
    [  0, 160, 160],   # teal     (spare)
], dtype=np.float32)
NOT_COUNTED = np.array([70, 70, 70], np.float32)
BOX_OUTLINE = np.array([255, 0, 0], np.uint8)
OUTLINE_WIDTH = 2            # outline thickness in pixels (drawing only -- areas are unaffected)
NAN_OUTLINE = np.array([150, 150, 150], np.uint8)   # skipped 'nan' boxes
FIELD_OUTLINE = np.array([200, 200, 200], np.uint8) # field boundary (layer 3)

CSV_HEADER = ["file", "row", "col", "date", "genotype", "in_plot_px", "canopy_px",
              "canopy_area_cm2", "segmentation_area_cm2", "canopy_area_m2", "segmentation_area_m2",
              "watershed_px", "watershed_area_cm2", "watershed_area_m2", "gain_m2",
              "cut_by_cap_px", "neighbours", "box_canopy_lost_px"]


# ----------------------------------------------------------------------
# inputs (same rules as predict_ortho.py / current-reclip_batch.py)
# ----------------------------------------------------------------------
def resolve_site_folder(site, date_str):
    for folder in SITE_FOLDERS.get(site, []):
        if (folder / "RGB" / date_str).is_dir():
            return folder
    return None


def pick_orthomosaic(date_dir):
    all_tifs = list(date_dir.glob("*.tif"))
    if not all_tifs:
        return None
    non_ms = [p for p in all_tifs if "mns" not in p.stem.lower() and "dsm" not in p.stem.lower()]
    cands = non_ms if non_ms else all_tifs
    return cands[0] if len(cands) == 1 else None


def pick_grid_shapefile(date_dir):
    cands = [p for p in date_dir.rglob("*.shp") if "bad" not in str(p).lower()]
    return cands[0] if len(cands) == 1 else None


def is_negated(site, date_str):
    return (site, date_str) in NEGATE or (site, "*") in NEGATE


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
    try:
        row_v, col_v = int(r[F_ROW]), int(r[F_COL])
    except Exception:
        return None
    raw = str(r[F_GENO]).strip()
    if raw.lower() in ("nan", "", "none"):
        geno = "nan"
    else:
        try:
            geno = str(int(float(raw)))
        except ValueError:
            geno = raw
    return f"r{row_v}_c{col_v}_{geno}_{site}_{date_tag}", row_v, col_v, geno


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def outline(mask, width=1):
    """Boundary pixels of a boolean mask, thickened to `width` pixels (drawing only)."""
    e = mask.copy()
    e[1:] &= mask[:-1]; e[:-1] &= mask[1:]
    e[:, 1:] &= mask[:, :-1]; e[:, :-1] &= mask[:, 1:]
    edge = mask & ~e
    if width > 1:
        edge = ndimage.binary_dilation(edge, iterations=width - 1)
    return edge


def box_frame(geom):
    """Centre, unit long axis, unit short axis, half length, half width of a (near-)rectangular box."""
    rect = geom.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    e1 = np.array([xs[1] - xs[0], ys[1] - ys[0]]); e2 = np.array([xs[2] - xs[1], ys[2] - ys[1]])
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    ax = e1 / l1 if l1 >= l2 else e2 / l2
    if ax[0] < 0 or (ax[0] == 0 and ax[1] < 0):
        ax = -ax
    c = np.array(rect.centroid.coords[0])
    return c, ax, np.array([-ax[1], ax[0]]), max(l1, l2) / 2, min(l1, l2) / 2


def directional_gaps(geoms):
    """Median gap end-to-end (along the box's long axis) and side-to-side (across it),
    from every box's nearest neighbour in that direction. Returns (along_m, across_m),
    either may be None if it can't be measured."""
    frames = [box_frame(g) for g in geoms]
    C = np.array([f[0] for f in frames])
    axes = np.array([f[1] for f in frames])
    ax = np.median(axes, 0); ax /= np.linalg.norm(ax)
    perp = np.array([-ax[1], ax[0]])
    HL = np.array([f[3] for f in frames]); HW = np.array([f[4] for f in frames])
    along, across = [], []
    for i in range(len(C)):
        d = C - C[i]
        a, b = d @ ax, d @ perp
        same_row = (np.abs(b) < HW[i]) & (a > 0.05)
        same_col = (np.abs(a) < HL[i]) & (b > 0.05)
        if same_row.any():
            j = np.argmin(np.where(same_row, a, np.inf)); along.append(a[j] - HL[i] - HL[j])
        if same_col.any():
            j = np.argmin(np.where(same_col, b, np.inf)); across.append(b[j] - HW[i] - HW[j])
    med = lambda v: float(np.median(v)) if len(v) >= 3 else None
    return med(along), med(across)


def median_gap_m(geoms):
    """Median distance from each box to its nearest neighbouring box (map units = metres)."""
    arr = np.array(geoms, dtype=object)
    gaps = []
    for i, g in enumerate(arr):
        d = shapely.distance(arr, g)
        d[i] = np.inf
        gaps.append(d.min())
    return float(np.median(gaps)), arr


def touching_pairs(lab):
    """Set of (a, b) label pairs that touch (4-neighbour), a < b, both > 0."""
    pairs = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            p = np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1)
            pairs.update(map(tuple, np.unique(p, axis=0)))
    return pairs


def assign_colours(idents, geom_arr, near_m, region_pairs):
    """Colour index per plot (1-based labels). Preferred colour (col + 2*row) % 5 gives
    different colours to all 8 grid neighbours; geometry + actual touching regions are
    checked too, and any clash is moved to the next free colour."""
    n = len(idents)
    adj = {k: set() for k in range(1, n + 1)}
    for i in range(n):
        d = shapely.distance(geom_arr, geom_arr[i])
        for j in np.nonzero(d <= near_m)[0]:
            if j != i:
                adj[i + 1].add(j + 1)
    for a, b in region_pairs:
        adj[a].add(b); adj[b].add(a)

    colour = {}
    for k in range(1, n + 1):
        _, row_v, col_v, _ = idents[k - 1]
        used = {colour[j] for j in adj[k] if j in colour}
        pref = (col_v + 2 * row_v) % 5
        if pref not in used:
            colour[k] = pref
        else:
            free = [c for c in range(len(PALETTE)) if c not in used]
            colour[k] = free[0] if free else pref
    return colour


def render(rgb, ws, counted_canopy_free, labels, colour, valid, nan_mask=None, field_mask=None):
    """Field overlay: plot regions coloured, uncounted canopy dark grey, box outlines white."""
    ov = rgb.astype(np.float32)
    lut = np.zeros((ws.max() + 1, 3), np.float32)
    for k, c in colour.items():
        lut[k] = PALETTE[c]
    reg = ws > 0
    ov[reg] = 0.35 * ov[reg] + 0.65 * lut[ws[reg]]
    ov[counted_canopy_free] = 0.4 * ov[counted_canopy_free] + 0.6 * NOT_COUNTED
    ov = ov.astype(np.uint8)
    ov[outline(labels > 0, OUTLINE_WIDTH)] = BOX_OUTLINE
    if nan_mask is not None:
        ov[outline(nan_mask, OUTLINE_WIDTH)] = NAN_OUTLINE
    if field_mask is not None:
        ov[outline(field_mask)] = FIELD_OUTLINE
    ov[~valid] = 0
    return ov


# ----------------------------------------------------------------------
def run_site_date(site, date_str):
    date_tag = f"{date_str[5:7]}-{date_str[8:10]}-{date_str[2:4]}"
    settings = {
        "THRESHOLD": THRESHOLD,
        "USE_TERRITORY_CAP": USE_TERRITORY_CAP,
        "REACH_FACTOR": REACH_FACTOR,
        "layer 1 SKIP_NAN_PLOTS": SKIP_NAN_PLOTS,
        "layer 2 NAN_COMPETE": NAN_COMPETE,
        "layer 3 FIELD_BOUNDARY": f"{FIELD_BOUNDARY} (margin {FIELD_MARGIN_FACTOR} x gap)",
        "layer 4 CORE_SEEDS": f"{CORE_SEEDS} (shrink {CORE_SHRINK} of short side)",
        "layer 4 DISTANCE_LANDSCAPE": DISTANCE_LANDSCAPE,
        "layer 5 DIRECTIONAL_REACH": DIRECTIONAL_REACH,
    }

    prob_path = PROB_ROOT / site / f"{site}_{date_str}_canopy_prob.tif"
    if not prob_path.exists():
        print(f"[{site} {date_str}] SKIP -- {prob_path.name} not found. Run predict_ortho.py first.")
        return None
    ortho, grid, problem = find_inputs(site, date_str)
    if problem:
        print(f"[{site} {date_str}] SKIP -- {problem}")
        return None
    print(f"\n[{site} {date_str}]  territory cap: {'ON' if USE_TERRITORY_CAP else 'OFF'}")

    # canopy probability (same grid predict_ortho.py used)
    with rasterio.open(prob_path) as p:
        prob = p.read(1)
        tf, crs, (H, W), bounds = p.transform, p.crs, p.shape, p.bounds
    gsd_x, gsd_y = abs(tf.a), abs(tf.e)
    px_area_cm2 = (gsd_x * 100) * (gsd_y * 100)

    # matching RGB + no-data mask
    with rasterio.open(ortho) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=win, boundless=True, fill_value=0, out_shape=(src.count, H, W))
    rgb = np.transpose(data[:3], (1, 2, 0)).copy()
    valid = data[3] > 0 if data.shape[0] >= 4 else np.ones((H, W), bool)

    # 1. canopy, never on no-data
    canopy = (prob >= round(THRESHOLD * 255)) & valid

    # plot boxes -> labels
    gdf = gpd.read_file(grid)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    idents, shapes, geoms = [], [], []
    all_geoms, nan_shapes = [], []
    all_idents = site_identities(gdf, site, date_tag)
    for i, (_, r) in enumerate(gdf.iterrows()):
        ident = all_idents[i]
        if ident is None or r.geometry is None or r.geometry.is_empty:
            continue
        all_geoms.append(r.geometry)              # spacing is measured on every box
        if SKIP_NAN_PLOTS and ident[3] == "nan":
            nan_shapes.append((r.geometry, 1))    # layer 1: not a seed, not reported
            continue
        idents.append(ident)
        geoms.append(r.geometry)
        shapes.append((r.geometry, len(idents)))
    labels = rasterize(shapes, out_shape=(H, W), transform=tf, fill=0, all_touched=False, dtype="int32")
    nan_mask = (rasterize(nan_shapes, out_shape=(H, W), transform=tf, fill=0, all_touched=False,
                          dtype="uint8") > 0) if nan_shapes else None
    n = len(idents)
    if SKIP_NAN_PLOTS:
        print(f"  layer 1: skipped {len(nan_shapes)} blank-genotype ('nan') boxes")

    gap_m, _ = median_gap_m(all_geoms)
    geom_arr = np.array(geoms, dtype=object)
    reach_m = REACH_FACTOR * gap_m
    reach_px = reach_m / gsd_x
    reach_along_m = reach_across_m = reach_m
    if DIRECTIONAL_REACH:
        g_al, g_ac = directional_gaps(all_geoms)
        reach_along_m = REACH_FACTOR * (g_al if g_al is not None else gap_m)
        reach_across_m = REACH_FACTOR * (g_ac if g_ac is not None else gap_m)
        reach_px = max(reach_along_m, reach_across_m) / gsd_x     # window padding only
        if USE_TERRITORY_CAP:
            print(f"  layer 5: directional reach -- along the row {reach_along_m:.2f} m, "
                  f"across the row {reach_across_m:.2f} m (was {reach_m:.2f} m both ways)")
    print(f"  {n} plots, GSD {gsd_x * 100:.2f} cm, median gap {gap_m:.2f} m"
          + (f" -> reach {reach_m:.2f} m ({reach_px:.0f} px)" if USE_TERRITORY_CAP else ""))

    # layer 2: nan boxes join the flood as "ghost" plots (labels n+1 ...) so they claim
    # the grass/weeds around them exactly as real plots would; dropped after the cap
    flood_labels = labels
    if NAN_COMPETE and SKIP_NAN_PLOTS and nan_shapes:
        ghosts = [(g, n + i + 1) for i, (g, _) in enumerate(nan_shapes)]
        flood_labels = rasterize(shapes + ghosts, out_shape=(H, W), transform=tf, fill=0,
                                 all_touched=False, dtype="int32")
    n_all = int(flood_labels.max()) if flood_labels.any() else n
    all_shapes = shapes + (ghosts if (NAN_COMPETE and SKIP_NAN_PLOTS and nan_shapes) else [])

    # layer 3: field boundary -- canopy outside the plot-grid outline can't be claimed
    field_mask = None
    flood_canopy = canopy
    if FIELD_BOUNDARY:
        field_geom = shapely.convex_hull(shapely.union_all(all_geoms)).buffer(FIELD_MARGIN_FACTOR * gap_m)
        field_mask = rasterize([(field_geom, 1)], out_shape=(H, W), transform=tf, fill=0,
                               all_touched=False, dtype="uint8") > 0
        flood_canopy = canopy & field_mask
        print(f"  layer 3: field boundary excludes {(canopy & ~field_mask).sum() * px_area_cm2 / 1e4:.1f} m2 "
              f"of canopy outside the plot grid")

    # 2. seeds   3./4. flood through canopy only
    if CORE_SEEDS:
        # layer 4: seed from each box's centre; fall back to the whole box if the centre is bare
        core_shapes = []
        for g, k in all_shapes:
            rect = g.minimum_rotated_rectangle
            xs, ys = rect.exterior.coords.xy
            sides = [np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2)]
            core = g.buffer(-CORE_SHRINK * min(sides))
            core_shapes.append((core if not core.is_empty else g, k))
        core_labels = rasterize(core_shapes, out_shape=(H, W), transform=tf, fill=0,
                                all_touched=False, dtype="int32")
        markers = np.where(flood_canopy, core_labels, 0)
        seeded = np.zeros(n_all + 1, bool)
        seeded[np.unique(markers)] = True
        for k in np.nonzero(~seeded[1:])[0] + 1:
            fb = (flood_labels == k) & flood_canopy
            markers[fb] = k
        n_fallback = int((~seeded[1:n + 1]).sum())
        print(f"  layer 4: core seeds ({n_fallback} plots had a bare centre -> whole box used)")
    else:
        markers = np.where(flood_canopy, flood_labels, 0)

    if DISTANCE_LANDSCAPE:
        # layer 4: thick canopy floods first, so plants split at their narrowest link
        landscape = -ndimage.distance_transform_edt(flood_canopy)
    else:
        landscape = np.zeros((H, W), np.float32)
    ws = watershed(landscape, markers=markers, mask=flood_canopy)


    # 5. territory cap: drop pixels farther than reach from the plot's OWN box
    cut_px = np.zeros(n_all + 1, np.int64)
    if USE_TERRITORY_CAP:
        pad = int(np.ceil(reach_px)) + 2
        objs = ndimage.find_objects(flood_labels)
        frames_by_label = {k: box_frame(g) for g, k in all_shapes} if DIRECTIONAL_REACH else {}
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            r0, r1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
            c0, c1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
            sub = ws[r0:r1, c0:c1]
            if DIRECTIONAL_REACH:
                # distance past the box END (along its long axis) and past its SIDE (across),
                # each scaled by its own reach; outside the rounded outline = beyond reach
                c, ax, pp, hl, hw = frames_by_label[k]
                rr, cc = np.mgrid[r0:r1, c0:c1]
                X = tf.c + (cc + 0.5) * tf.a + (rr + 0.5) * tf.b
                Y = tf.f + (cc + 0.5) * tf.d + (rr + 0.5) * tf.e
                a = (X - c[0]) * ax[0] + (Y - c[1]) * ax[1]
                b = (X - c[0]) * pp[0] + (Y - c[1]) * pp[1]
                da = np.maximum(np.abs(a) - hl, 0) / reach_along_m
                db = np.maximum(np.abs(b) - hw, 0) / reach_across_m
                beyond = (da ** 2 + db ** 2) > 1
            else:
                box = flood_labels[r0:r1, c0:c1] == k
                beyond = ndimage.distance_transform_edt(~box) > reach_px
            drop = (sub == k) & beyond
            cut_px[k] = int(drop.sum())
            sub[drop] = 0
        # anything of plot k left outside its padded window is beyond reach too
        far = ws > 0
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            r0, r1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
            c0, c1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
            far[r0:r1, c0:c1] &= ws[r0:r1, c0:c1] != k
        stray = far & (ws > 0)
        if stray.any():
            cut_px += np.bincount(ws[stray], minlength=n_all + 1)[:n_all + 1]
            ws[stray] = 0

    # layer 2: throw away what the nan "ghosts" claimed -- nobody gets it
    if n_all > n:
        ghost_px = ws > n
        print(f"  layer 2: nan boxes competed and held {ghost_px.sum() * px_area_cm2 / 1e4:.2f} m2 "
              f"(not counted for any plot)")
        ws[ghost_px] = 0
    cut_px = cut_px[:n + 1]

    # box canopy that no real plot ended up with (cut off from the centre by a gap,
    # dropped by a neighbour's cap, or taken by a nan ghost) stays with the box owner --
    # only a real neighbour that keeps it can take it away
    orphan = canopy & (ws == 0) & (labels > 0)
    ws[orphan] = labels[orphan]

    # box canopy taken by a neighbour (only possible with CORE_SEEDS)
    lost_mask = (labels > 0) & canopy & (ws > 0) & (ws != labels)
    lost_px = np.bincount(labels[lost_mask], minlength=n + 1)[:n + 1]

    # per-plot numbers
    in_plot_px = np.bincount(labels.ravel(), minlength=n + 1)
    box_canopy_px = np.bincount(labels.ravel(), weights=canopy.ravel(), minlength=n + 1).astype(int)
    ws_px = np.bincount(ws.ravel(), minlength=n + 1)

    region_pairs = touching_pairs(ws)
    neigh = {k: [] for k in range(1, n + 1)}
    for a, b in region_pairs:
        neigh[a].append(b); neigh[b].append(a)

    rows = []
    for k, (name, row_v, col_v, geno) in enumerate(idents, start=1):
        ip, bp, wp = int(in_plot_px[k]), int(box_canopy_px[k]), int(ws_px[k])
        if ip == 0:
            continue
        b_cm2, s_cm2, w_cm2 = bp * px_area_cm2, ip * px_area_cm2, wp * px_area_cm2
        nb = ";".join(f"r{idents[j - 1][1]}_c{idents[j - 1][2]}" for j in sorted(neigh[k]))
        rows.append((name, row_v, col_v, date_str, geno, ip, bp,
                     round(b_cm2, 1), round(s_cm2, 1), round(b_cm2 / 1e4, 4), round(s_cm2 / 1e4, 4),
                     wp, round(w_cm2, 1), round(w_cm2 / 1e4, 4), round((w_cm2 - b_cm2) / 1e4, 4),
                     int(cut_px[k]), nb, int(lost_px[k])))
    rows.sort(key=lambda r: (r[1], r[2]))

    out_dir = OUT_ROOT / site
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{site}_{date_str}_watershed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    # handoff CSV: canopy_* = watershed canopy assigned to the plot (incl. beyond the box)
    if WRITE_HANDOFF:
        HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
        with open(HANDOFF_DIR / f"{site}_{date_str}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HANDOFF_HEADER)
            w.writerows([r[0:5] + (r[12], r[13]) for r in rows])

    # checks + summary
    box_m2 = np.array([r[9] for r in rows]); ws_m2 = np.array([r[13] for r in rows])
    never_less = int((ws_m2 + 1e-9 >= box_m2).sum())
    total_canopy_m2 = canopy.sum() * px_area_cm2 / 1e4
    not_counted_m2 = (canopy & (ws == 0)).sum() * px_area_cm2 / 1e4
    print(f"  wrote {csv_path.name} ({len(rows)} plots)")
    from datetime import datetime
    with open(out_dir / f"{site}_{date_str}_run_settings.txt", "w") as f:
        f.write(f"run: {datetime.now():%Y-%m-%d %H:%M}\n")
        if DIRECTIONAL_REACH:
            f.write(f"reach: along row {reach_along_m:.2f} m, across row {reach_across_m:.2f} m "
                    f"(median nearest gap {gap_m:.2f} m)\n")
        else:
            f.write(f"reach: {reach_m:.2f} m (median gap {gap_m:.2f} m)\n")
        for k, v in settings.items():
            f.write(f"{k}: {v}\n")
    print(f"  mean box area {box_m2.mean():.3f} m2 -> watershed {ws_m2.mean():.3f} m2 "
          f"(mean gain {(ws_m2 - box_m2).mean():+.3f} m2, {((ws_m2 / np.maximum(box_m2, 1e-9)) - 1)[box_m2 > 0].mean() * 100:+.1f}%)")
    print(f"  plots touching a neighbour : {sum(1 for r in rows if r[16])}/{len(rows)}")
    if USE_TERRITORY_CAP:
        print(f"  plots cut by the cap       : {sum(1 for r in rows if r[15] > 0)}/{len(rows)}")
    if CORE_SEEDS:
        print(f"  plots that lost box canopy to a neighbour: {sum(1 for r in rows if r[17] > 0)}/{len(rows)}"
              f"  (watershed >= box for {never_less}/{len(rows)} -- not guaranteed with core seeds)")
    else:
        print(f"  check: watershed >= box for {never_less}/{len(rows)} plots (should be all)")
    print(f"  check: assigned {ws_m2.sum():.1f} m2 of {total_canopy_m2:.1f} m2 canopy in view "
          f"({not_counted_m2:.1f} m2 not counted: borders, weeds, beyond cap) -- nothing counted twice")

    # overlays
    if SAVE_FIELD_OVERLAY or SAVE_PLOT_OVERLAYS or WRITE_HANDOFF:
        colour = assign_colours(idents, geom_arr, 2.5 * gap_m, region_pairs)
        ov = render(rgb, ws, canopy & (ws == 0), labels, colour, valid, nan_mask, field_mask)
        if SAVE_FIELD_OVERLAY or WRITE_HANDOFF:
            img = Image.fromarray(ov)
            if max(img.size) > OVERLAY_MAX_SIDE:
                s = OVERLAY_MAX_SIDE / max(img.size)
                img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.NEAREST)
            if SAVE_FIELD_OVERLAY:
                img.save(out_dir / f"{site}_{date_str}_watershed_overlay.png")
            if WRITE_HANDOFF:
                img.save(HANDOFF_DIR / f"{site}_{date_str}.png")
        if SAVE_PLOT_OVERLAYS:
            pdir = out_dir / "overlays"
            pdir.mkdir(exist_ok=True)
            pad = int(np.ceil(reach_px)) + 4
            for k, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is None:
                    continue
                r0, r1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
                c0, c1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
                crop = ov[r0:r1, c0:c1].copy()
                own = outline(labels[r0:r1, c0:c1] == k, OUTLINE_WIDTH)
                crop[own] = [255, 0, 0]                       # this plot's box in red (others are white)
                h, w_ = crop.shape[:2]
                Image.fromarray(crop).resize((w_ * 4, h * 4), Image.NEAREST).save(
                    pdir / f"{idents[k - 1][0]}.png")
    return len(rows)


def main():
    summary = []
    for site, dates in SITE_DATES.items():
        for d in dates:
            if is_negated(site, d):
                print(f"[{site} {d}] NEGATED -- skipped (see NEGATE in CONFIG)")
                continue
            try:
                n = run_site_date(site, d)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[{site} {d}] FAILED: {e}")
                n = None
            summary.append((site, d, n))
    print("\n===== SUMMARY =====")
    for site, d, n in summary:
        print(f"  {site:8s} {d}   {f'{n} plots' if n is not None else 'skipped / failed'}   -> {OUT_ROOT / site}")


if __name__ == "__main__":
    main()
