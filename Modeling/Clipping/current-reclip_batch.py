"""
reclip_batch.py
----------------
Generalized version of Reclip_script_singeldate.py. Builds ONE clean clip
archive across every site and date listed in the master manifest, instead of
one hardcoded site/date at a time.

For each unique (Site, Date) in the manifest's "Combined" sheet:
  * find that site/date's raw RGB orthomosaic and GRID shapefile on disk
  * re-clip every polygon (skip blank/nan genotype boxes)
  * write r{row}_c{col}_{var}_{Site}_{date}.tif into a fresh archive folder,
    with georeferencing (CRS + transform) preserved

No manifest.csv / sidecar files are written. Every decision (which files were
picked, what was skipped, what's ambiguous) is printed to the terminal only,
so this is safe to re-run: it never overwrites its own output silently and
never leaves extra bookkeeping files behind.

Re-runnable: a plot whose output .tif already exists is skipped, so a partial
run (network hiccup, Ctrl-C, etc.) can just be started again.

Row/col/var correctness verified against the site's independent field maps
(GodetC1_map.csv, RoujolA_map.csv, RoujolD_gv.csv) across every date for
RoujolA, RoujolD, and GodetC1 -- 100% match, no row/col swap needed.

Godet25 is different: its shapefiles only ever carry a flat plot `id` (no
row/col at all). Godet25_map.csv (uploaded as "Godet26.csv") is a 25x33
staggered field map of genotype ids that DOES encode row/col, but it has no
join key to the shapefile -- so row/col is reconstructed geometrically per
date: reproject to UTM, find the field's rotation from its minimum bounding
rectangle, rotate so field-rows align to an axis, band polygons into 25 rows
by the 24 largest along-axis gaps, then match sort order within each row
against the map's column layout. Verified across all 22 Godet25 manifest
dates: 20 dates at 406/406 id-agreement, 2 dates (2025-10-16, 2026-01-20)
needed the mirrored orientation and also hit 406/406, and the 9 earliest
dates (06-03 through 07-29 2025) sit at 405/406 -- one consistent single-cell
disagreement (map row 9 col 33: map says 223, shapefile says 233) that looks
like a one-digit transcription typo, not a reconstruction error. Because the
genotype label always comes from the shapefile's own `id` field (never from
the map), that one disagreement can never produce a mislabeled clip -- the
map is only used to derive row/col position.

Requires: pip install geopandas rasterio
Run in the yam-fcover env.
"""

import csv
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import MultiPoint

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# EDIT if needed: the lab "Yam/DroneData/Orthophotos" folder.
# Default = the Shakoor Lab OneDrive path on a Mac, for whoever is logged in.
ROOT = Path.home() / "Library/CloudStorage/OneDrive-DDPSC/Shakoor Lab (Notes)/Yam/DroneData/Orthophotos"
PHOTOS = ROOT / "Photos"
MANIFEST = ROOT / "Yam-fcover" / "1RGB_MasterDataset_Split_By_Site.xlsx"
OUT_ROOT = ROOT / "Yam-fcover" / "clip_archive"
GODET25_MAP = Path(__file__).resolve().parent / "Godet25_map.csv"   # kept next to this script

SITE_FOLDERS = {
    "RoujolA": [PHOTOS / "2024-25" / "RoujolA"],
    "RoujolD": [PHOTOS / "2024-25" / "RoujolD"],
    "GodetC1": [PHOTOS / "2024-25" / "GodetC1", PHOTOS / "2024-25_1" / "GodetC1_2"],
    "Godet25": [PHOTOS / "2025-26_part1_extracted" / "godet_2025"],
}

SITES_TO_RUN = ["RoujolA", "RoujolD", "GodetC1", "Godet25"]

F_ROW, F_COL, F_GENO = "row", "col", "var"
GODET25_EXPECTED_POLYGONS = 406
GODET25_NUM_ROWS = 25

# ----------------------------------------------------------------------
def pick_orthomosaic(date_dir: Path):
    """Top-level .tif in the date folder. 'mns'/'dsm' (surface-model exports)
    are only used as a TIEBREAKER when there's more than one candidate --
    some dates only ever got one tif exported and it happens to have 'mns'
    in its name too, so excluding it outright would wrongly call a real
    orthomosaic 'missing'. Returns (path, warning); path is None if 0 or
    >1 candidates remain after the tiebreak."""
    all_tifs = list(date_dir.glob("*.tif"))
    if len(all_tifs) == 0:
        return None, "NO orthomosaic tif found"
    non_ms = [p for p in all_tifs if "mns" not in p.stem.lower() and "dsm" not in p.stem.lower()]
    cands = non_ms if non_ms else all_tifs
    if len(cands) > 1:
        return None, f"AMBIGUOUS orthomosaic ({len(cands)} candidates): {[c.name for c in cands]}"
    warn = None
    if cands is all_tifs and "mns" in cands[0].stem.lower():
        warn = f"NOTE: only candidate is mns-named ({cands[0].name}), using it anyway"
    return cands[0], warn


def pick_grid_shapefile(date_dir: Path):
    """Any *.shp under the date folder whose path doesn't contain 'bad'.
    Returns (path, warning) -- path is None if 0 or >1 candidates remain."""
    cands = [p for p in date_dir.rglob("*.shp") if "bad" not in str(p).lower()]
    if len(cands) == 0:
        return None, "NO grid shapefile found"
    if len(cands) > 1:
        return None, f"AMBIGUOUS grid shapefile ({len(cands)} candidates): {[c.name for c in cands]}"
    return cands[0], None


def resolve_site_folder(site: str, date_str: str):
    """Return the physical site folder that actually has RGB/<date_str>."""
    for folder in SITE_FOLDERS.get(site, []):
        if (folder / "RGB" / date_str).is_dir():
            return folder
    return None


# ----------------------------------------------------------------------
# Godet25-only: reconstruct row/col from geometry + the field map, since the
# shapefile itself only has id/geometry.
# ----------------------------------------------------------------------
def load_godet25_map(path: Path):
    """Parse the staggered field map into per-csv-row lists of (col, genotype),
    1-indexed, ascending column order, blanks dropped."""
    with open(path, encoding="utf-8-sig") as f:
        csv_rows = list(csv.reader(f))
    row_entries = []
    for r in csv_rows:
        entries = [(c_idx, int(v.strip())) for c_idx, v in enumerate(r, start=1) if v.strip() != ""]
        row_entries.append(entries)
    return row_entries


def assign_godet25_row_col(gdf, row_entries):
    """Return (row_col_array, match, total, note) where row_col_array is a
    list of (row, col) aligned to gdf's original row order, or None if no
    orientation reconstructs the field correctly for this date."""
    if len(gdf) != GODET25_EXPECTED_POLYGONS:
        return None, 0, 0, f"unexpected polygon count {len(gdf)} (expected {GODET25_EXPECTED_POLYGONS})"

    expected_sizes = [len(e) for e in row_entries]
    utm = gdf.estimate_utm_crs()
    g = gdf.to_crs(utm)
    cx = g.geometry.centroid.x.values
    cy = g.geometry.centroid.y.values

    pts = MultiPoint(list(zip(cx, cy)))
    coords = list(pts.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % 4]
        edges.append((math.hypot(x1 - x0, y1 - y0), math.degrees(math.atan2(y1 - y0, x1 - x0))))
    edges.sort(key=lambda t: -t[0])
    theta = math.radians(-edges[0][1])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rx = cx * cos_t - cy * sin_t
    ry = cx * sin_t + cy * cos_t

    order = np.argsort(ry)
    ry_sorted = ry[order]
    gaps = np.diff(ry_sorted)
    n_cuts = GODET25_NUM_ROWS - 1
    cut_positions = np.sort(np.argsort(gaps)[::-1][:n_cuts])
    boundaries = []
    prev = 0
    for cp in cut_positions:
        boundaries.append((prev, cp))
        prev = cp + 1
    boundaries.append((prev, len(ry_sorted) - 1))
    if len(boundaries) != GODET25_NUM_ROWS:
        return None, 0, 0, f"could not split into {GODET25_NUM_ROWS} row-bands (got {len(boundaries)})"

    sizes = [b - a + 1 for a, b in boundaries]
    band = np.zeros(len(g), dtype=int)
    for band_num, (a, b) in enumerate(boundaries):
        band[order[a:b + 1]] = band_num

    best = None
    for row_desc in (False, True):
        expected_seq = [expected_sizes[(GODET25_NUM_ROWS - 1 - b) if row_desc else b] for b in range(GODET25_NUM_ROWS)]
        if sizes != expected_seq:
            continue
        for col_desc in (False, True):
            assignment = [None] * len(g)
            match = total = 0
            for band_num in range(GODET25_NUM_ROWS):
                idxs_in_band = np.where(band == band_num)[0]
                idxs_in_band = idxs_in_band[np.argsort(rx[idxs_in_band] * (-1 if col_desc else 1))]
                csv_row_idx = (GODET25_NUM_ROWS - 1 - band_num) if row_desc else band_num
                entries = row_entries[csv_row_idx]
                for (col_idx, expected_val), poly_pos in zip(entries, idxs_in_band):
                    total += 1
                    actual_id = g.iloc[poly_pos]["id"]
                    actual_id = None if pd.isna(actual_id) else int(float(actual_id))
                    if actual_id == expected_val:
                        match += 1
                    assignment[poly_pos] = (csv_row_idx + 1, col_idx)
            if best is None or match > best[0]:
                best = (match, total, assignment)
    if best is None:
        return None, 0, 0, "no row/col orientation matched this date's row-size pattern"
    match, total, assignment = best
    return assignment, match, total, None


# ----------------------------------------------------------------------
def main():
    df = pd.read_excel(MANIFEST, sheet_name="Combined")
    df = df[df["Site"].isin(SITES_TO_RUN)]
    pairs = sorted(df[["Site", "Date"]].drop_duplicates().itertuples(index=False), key=lambda r: (r.Site, r.Date))
    print(f"manifest: {len(df)} valid rows (sites={SITES_TO_RUN}) -> {len(pairs)} unique (Site, Date) pairs to re-clip\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    godet25_map = None
    if "Godet25" in SITES_TO_RUN:
        godet25_map = load_godet25_map(GODET25_MAP)

    totals = dict(written=0, skipped_existing=0, skipped_blank_geno=0, skipped_bad_record=0,
                  dates_ok=0, dates_skipped=0)
    skipped_dates = []

    for site, date in pairs:
        date_str = date.strftime("%Y-%m-%d")
        date_tag = date.strftime("%m-%d-%y")

        site_folder = resolve_site_folder(site, date_str)
        if site_folder is None:
            print(f"[{site} {date_str}] SKIP -- no on-disk RGB/{date_str} folder found for this site")
            skipped_dates.append((site, date_str, "no matching site folder"))
            totals["dates_skipped"] += 1
            continue

        date_dir = site_folder / "RGB" / date_str
        ortho, ortho_warn = pick_orthomosaic(date_dir)
        grid, grid_warn = pick_grid_shapefile(date_dir)

        if ortho is None or grid is None:
            print(f"[{site} {date_str}] SKIP -- {ortho_warn or 'ortho OK'} | {grid_warn or 'grid OK'}")
            skipped_dates.append((site, date_str, ortho_warn or grid_warn))
            totals["dates_skipped"] += 1
            continue
        if ortho_warn:
            print(f"[{site} {date_str}] {ortho_warn}")
        if grid_warn:
            print(f"[{site} {date_str}] {grid_warn}")

        out_dir = OUT_ROOT / site / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(grid)
        except Exception as e:
            print(f"[{site} {date_str}] SKIP -- could not read shapefile {grid.name}: {e}")
            skipped_dates.append((site, date_str, f"shapefile read error: {e}"))
            totals["dates_skipped"] += 1
            continue

        godet25_assignment = None
        if site == "Godet25":
            godet25_assignment, m, t, note = assign_godet25_row_col(gdf, godet25_map)
            if godet25_assignment is None:
                print(f"[{site} {date_str}] SKIP -- Godet25 row/col reconstruction failed: {note}")
                skipped_dates.append((site, date_str, f"row/col reconstruction failed: {note}"))
                totals["dates_skipped"] += 1
                continue
            print(f"[{site} {date_str}] row/col reconstructed from field map: {m}/{t} id-agreement "
                  f"({100*m/t:.1f}%) -- genotype labels always taken from the shapefile's own id, not the map")
        else:
            missing = [c for c in (F_ROW, F_COL, F_GENO) if c not in gdf.columns]
            if missing:
                print(f"[{site} {date_str}] SKIP -- shapefile is missing column(s) {missing} "
                      f"(has: {sorted(gdf.columns)}); this site/date needs its own naming scheme")
                skipped_dates.append((site, date_str, f"missing schema columns: {missing}"))
                totals["dates_skipped"] += 1
                continue

        written = skipped_blank = skipped_existing = skipped_bad = 0

        with rasterio.open(ortho) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            for i, (_, r) in enumerate(gdf.iterrows()):
                try:
                    if site == "Godet25":
                        rc = godet25_assignment[i]
                        if rc is None:
                            skipped_bad += 1
                            continue
                        row_v, col_v = rc
                        raw_geno = str(r["id"]).strip()
                    else:
                        row_v = int(r[F_ROW])
                        col_v = int(r[F_COL])
                        raw_geno = str(r[F_GENO]).strip()
                except Exception:
                    skipped_bad += 1
                    continue

                if raw_geno.lower() in ("nan", "", "none"):
                    skipped_blank += 1
                    continue
                try:
                    geno = str(int(float(raw_geno)))
                except ValueError:
                    skipped_bad += 1
                    continue

                name = f"r{row_v}_c{col_v}_{geno}_{site}_{date_tag}.tif"
                out_path = out_dir / name
                if out_path.exists():
                    skipped_existing += 1
                    continue

                try:
                    out_img, out_tf = mask(src, [r.geometry], crop=True)
                except Exception:
                    skipped_bad += 1
                    continue

                meta = src.meta.copy()
                meta.update(height=out_img.shape[1], width=out_img.shape[2], transform=out_tf)
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(out_img)
                written += 1

        totals["written"] += written
        totals["skipped_existing"] += skipped_existing
        totals["skipped_blank_geno"] += skipped_blank
        totals["skipped_bad_record"] += skipped_bad
        totals["dates_ok"] += 1

        print(f"[{site} {date_str}] ortho={ortho.name}  grid={grid.parent.name}/{grid.name}  "
              f"-> wrote {written}, skipped {skipped_existing} existing, "
              f"{skipped_blank} blank-genotype, {skipped_bad} bad records")

    print("\n" + "=" * 70)
    print(f"dates processed OK : {totals['dates_ok']}")
    print(f"dates skipped      : {totals['dates_skipped']}")
    print(f"clips written      : {totals['written']}")
    print(f"clips already there: {totals['skipped_existing']}")
    print(f"blank-genotype skip: {totals['skipped_blank_geno']}")
    print(f"bad-record skip    : {totals['skipped_bad_record']}")
    if skipped_dates:
        print(f"\nSKIPPED DATES ({len(skipped_dates)}) -- needs a manual look:")
        for site, date_str, reason in skipped_dates:
            print(f"  {site:10s} {date_str:12s} {reason}")
    print(f"\narchive root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
