# Plot Vegetation Index Extraction

Jupyter notebooks that calculate plot-level vegetation indices from drone orthomosaics (RGB and multispectral) and a plot grid shapefile.

## In short

You give the notebooks a drone image of a field trial and a map of the plot boundaries. For each plot, the notebooks calculate the mean value of each image band and a set of vegetation indices. The result is a CSV table with one row for each plot.

There are two types of notebook:

- **Template notebooks** process one image (one flight date). Use them to learn the method or to process a single image.
- **Batch notebooks** process all flight dates for one site in one run. Use them for most work.
- **Supplemental notebooks** join the batch CSVs into larger tables: one table for each site, one table for all sites, and an Excel workbook. Use them after the batch notebooks.

The batch notebooks do not need the template notebooks to run.

## Contents

```
Original_Template_Script/
  Generate_RGB indices_yam.ipynb       one RGB image  -> one CSV
  Generate_MS indices_yam.ipynb        one MS image   -> one CSV
VI_Batch/
  Batch_generate_rgb_indicies.ipynb    all RGB dates for one site -> one CSV for each date
  Batch_generate_MS_indicies.ipynb     all MS dates for one site  -> one CSV for each date
Supplemental/
  1_merge_dates_by_site.ipynb          all date CSVs for one site -> one site master CSV
  2_combine_sites.ipynb                site master CSVs           -> one all-sites master CSV
  3_split_by_site_workbook.ipynb       all-sites master CSV       -> Excel workbook
```

| Notebook | Input | Output |
|---|---|---|
| Template RGB | 1 RGB `.tif` and 1 `.shp` | 1 CSV, 3 band means and 25 indices |
| Template MS | 1 MS `.tif` and 1 `.shp` | 1 CSV, 4 band means and 46 indices |
| Batch RGB | 1 site folder | 1 CSV for each date and 1 summary CSV |
| Batch MS | 1 site folder | 1 CSV for each date and 1 summary CSV |
| 1 Merge dates | the date CSVs for 1 site | 1 site master CSV |
| 2 Combine sites | 2 or more site master CSVs | 1 all-sites master CSV |
| 3 Split workbook | 1 all-sites master CSV | 1 Excel workbook, 1 sheet for each site |

Some file names contain spaces. Put the path in quotes when you use it in a terminal.

## Requirements

- Python 3.9 or later
- `numpy`, `pandas`, `geopandas`, `rasterio`, `jupyter`
- `tqdm` (shows a progress bar, optional in the batch notebooks, necessary in `3_split_by_site_workbook.ipynb`)
- `xlsxwriter` (only for `3_split_by_site_workbook.ipynb`)

Install with conda. This is the recommended method, because `rasterio` and `geopandas` need GDAL:

```bash
conda create -n vi-extract -c conda-forge python=3.11 numpy pandas geopandas rasterio tqdm xlsxwriter jupyter
conda activate vi-extract
```

Or install with pip:

```bash
pip install numpy pandas geopandas rasterio tqdm xlsxwriter jupyter
```

## Input data

### Raster band order

| Image | Band 1 | Band 2 | Band 3 | Band 4 | Band 5 |
|---|---|---|---|---|---|
| RGB | Red | Green | Blue | | |
| MS | Red | Green | NIR | RedEdge | Alpha (not used) |

If your raster has a different band order, the indices will be wrong. Check the band order before you run a notebook.

**RGB value scale:** the RGB indices assume 8-bit values (0 to 255). CIVE and VVI use fixed constants for this scale, so if your RGB orthomosaics use a different range (for example 16-bit, 0 to 65535), convert the bands to 0 to 255 first or do not use these two indices.

### MS radiometric scale

The MS orthomosaics must contain reflectance values (0 to 1). Before you run a notebook, check the band value range of one file (for example, in QGIS).

- If the values are between 0 and 1 (reflectance), use the notebooks as they are.
- If the values are much larger (for example 0 to 65535, raw pixel values), convert the bands to reflectance before you calculate the indices. If you cannot convert them, do not use the indices that add a fixed constant: SAVI, MSAVI, OSAVI, OSAVI_2, TSAVI, PVI, SAVI_2, ATSAVI, EVI_2, MEVI, MCARI_2, MTVI_2, R_MCARI_MTVI_2, R_TCARI_OSAVI, MCARI_OSAVI and TCARI_OSAVI. Their constants (for example 0.5, 0.16 or 1.0) assume reflectance, so with raw values their results have no meaning.

Normalized difference and ratio indices (for example NDVI, gNDVI, NDRE and RVI_1) do not change with the value scale.

### Plot grid shapefile

- The shapefile has one polygon for each plot.
- **Plot ID column:** the template notebooks use the first column (`PLOT_ID_INDEX = 0`). The batch notebooks use the first column they find from this list: `var`, `id`, `plot_id`, `Plot_ID`, `plot`, `Plot`, `name`, `Name`.
- **Coordinate reference system (CRS):** for the template notebooks, the shapefile and the raster must use the same CRS. The batch notebooks change the shapefile to the raster CRS if they are different.

### Folder layout (batch notebooks only)

The batch notebooks find the files from the folder layout. Use this layout:

```
site_folder/
  RGB/
    2024-11-20/
      orthomosaic.tif
      plot_grid.shp          (can be in a subfolder)
    2024-12-11/
      ...
  MS/
    2024-11-20/
      orthomosaic.tif
    2024-12-11/
      ...
```

- Use the format `YYYY-MM-DD` for all date folders (RGB and MS). The MS notebook finds the plot grid in the RGB folder with the same name, so both folders must use the same date format.
- The MS batch notebook uses the plot grid from `RGB/<same date>/`. If it finds no grid there, it looks in `MS/<date>/`.
- A date folder can contain more than one `.tif` or `.shp`. In that case, the notebook selects one file. It prefers shapefiles with "grid" in the path, and it puts paths with "bad" last. The summary CSV shows the files that the notebook used.

## Quick start

### Batch notebook (all dates for one site)

1. Open `VI_Batch/Batch_generate_rgb_indicies.ipynb` or `VI_Batch/Batch_generate_MS_indicies.ipynb`.
2. Go to the `USER SETTINGS` block at the end of the code cell.
3. Set the paths:

   ```python
   site_folder = 'path/to/site_folder'
   output_root = 'path/to/output'        # RGB notebook
   output_dir = 'path/to/output/MS'      # MS notebook
   dates_to_process = None               # or ["2024-11-20", "2024-12-11"]
   ```

4. Run the cell.
5. Open the summary CSV. Look for dates with a status other than `processed`.

Test on one date first. Set `dates_to_process = ["YYYY-MM-DD"]`, run the cell, and check the CSV. Then set `dates_to_process = None` to process all dates.

To join the date CSVs into one table, use the [supplemental notebooks](#supplemental-notebooks).

### Template notebook (one image)

1. Open `Original_Template_Script/Generate_RGB indices_yam.ipynb` or `Original_Template_Script/Generate_MS indices_yam.ipynb`.
2. Set the three paths at the top of the code cell:

   ```python
   ms_tif_path = 'path/to/MS/YYYY-MM-DD/ms_orthomosaic.tif'   # rgb_tif_path in the RGB notebook
   shp_path = 'path/to/RGB/YYYY-MM-DD/plot_grid.shp'
   out_csv_path = 'path/to/output/ms_vi.csv'
   ```

3. Run the cell.

## Output

### Plot CSV

Each CSV has one row for each plot polygon. The columns are:

1. All attribute columns from the shapefile, and the polygon geometry as text (WKT).
2. The mean value of each band.
3. The vegetation indices.

| Notebook | Output file |
|---|---|
| Template | the path in `out_csv_path` |
| Batch RGB | `output_root/<site>/<site>_<YYYY-MM-DD>_RGB_VI.csv` |
| Batch MS | `output_dir/<site>_<YYYY-MM-DD>_MS_VI.csv` |

`<site>` is the name of `site_folder`.

The plot grid shapefile can have a column with the same name as an index (for example `NDVI`). In that case, the CSV has two columns with that name. The first column comes from the shapefile. The second column is the calculated index. pandas reads the second column as `NDVI.1`.

The batch notebooks overwrite a CSV that has the same name. To keep existing files, add `overwrite=False` to the `run_batch_rgb_vi(...)` or `run_batch_ms_vi(...)` call.

### Summary CSV (batch notebooks only)

The batch notebooks also write `<site>_RGB_VI_batch_summary.csv` or `<site>_MS_VI_batch_summary.csv`. This file has one row for each date. It shows the raster, the shapefile, the number of plots, the number of warnings, and the status.

| Status | Meaning |
|---|---|
| `processed` | The CSV was written. |
| `skipped_missing_rgb_tif` / `skipped_missing_ms_tif` | The date folder has no `.tif` file. |
| `skipped_missing_shapefile` | The notebook found no plot grid for this date. |
| `skipped_existing_output` | The CSV exists and `overwrite=False`. |
| `failed` | An error occurred. The `message` column shows the error. |

If the calculation fails for one plot, the batch notebook writes `NaN` for that plot and adds one warning. The run continues with the next plot.

### Vegetation indices

<details>
<summary>RGB: 3 band means and 25 indices</summary>

`Mean Blue`, `Mean Green`, `Mean Red`, `RCC`, `GCC`, `ExR`, `ExG`, `ExGR`, `VARI_2`, `MGRVI`, `CIVE`, `VEG`, `RGVI`, `NPCI`, `SRPI`, `NGRDI`, `SCI`, `GLI`, `HI`, `VARI`, `GLAI`, `CI`, `SHP`, `SAT`, `HUE`, `RI`, `BI`, `VVI`

</details>

<details>
<summary>MS: 4 band means and 46 indices</summary>

`Mean Green`, `Mean Red`, `Mean RedEdge`, `Mean NIR`, `NDVI`, `gNDVI`, `RVI_1`, `GCI`, `RGVI`, `DVI`, `SAVI`, `MSAVI`, `OSAVI`, `RDVI`, `TVI`, `TSAVI`, `PVI`, `SAVI_2`, `ATSAVI`, `NDWI`, `RVI_2`, `MCARI`, `MCARI_1`, `MCARI_2`, `MTVI_1`, `MTVI_2`, `R_MCARI_MTVI_2`, `DATT`, `NDCI`, `PSRI`, `PSRI2`, `PSRI3`, `SPVI`, `TCARI`, `R_TCARI_OSAVI`, `RERI`, `NDRE`, `MTCI`, `EVI_2`, `RECI`, `NGRDI`, `ARI_2`, `CRI_2`, `NRVI`, `OSAVI_2`, `MEVI`, `TVI_2`, `MCARI_OSAVI`, `TCARI_OSAVI`, `WDRVI`

</details>

The formulas are in the `extract_by_mask` function (template notebooks and MS batch notebook) and the `calculate_rgb_indices` function (RGB batch notebook).

## Supplemental notebooks

The supplemental notebooks join the CSVs from the batch notebooks into larger tables. They work with RGB data and MS data. Run them separately for each image type. Do not combine RGB files and MS files in one run.

```
Batch notebook                  one CSV for each date
  1_merge_dates_by_site         one master CSV for each site
  2_combine_sites               one master CSV for all sites
  3_split_by_site_workbook      one Excel workbook (optional)
```

Each notebook has a `USER SETTINGS` block at the top of the code cell. Set the paths there. Then run the cell.

### 1. Merge dates for one site

`Supplemental/1_merge_dates_by_site.ipynb` joins all the date CSVs of one site into one table. Run it one time for each site and each image type.

| Setting | Value |
|---|---|
| `input_folder` | The folder with the date CSVs from the batch notebook |
| `output_folder` | The folder for the site master CSV |
| `image_type_label` | `"MS"` or `"RGB"` |
| `site_name_override` | Optional. A site name to use instead of the name in the file names. |

Output: `output_folder/<site>_<MS or RGB>_master.csv`

The notebook adds these columns at the start of the table:

| Column | Value |
|---|---|
| `Site` | The site from the file name, or `site_name_override` |
| `Date` | The date from the file name (`YYYY-MM-DD`) |
| `Image_Type` | `MS` or `RGB` |
| `Missing_val` | The number of empty values in that row |

Use `site_name_override` to make the site name the same in the RGB data and the MS data.

### 2. Combine sites

`Supplemental/2_combine_sites.ipynb` joins the site master CSVs into one table for all sites.

| Setting | Value |
|---|---|
| `image_type_label` | `"MS"` or `"RGB"`. Used in the output file name. |
| `master_files` | The list of site master CSVs |
| `output_folder` | The folder for the all-sites master CSV |

Output: `output_folder/<MS or RGB>_MasterDataset_AllSites.csv`

- The first file in `master_files` sets the column order. If another file has different columns, the notebook prints the differences. A missing column is added with empty values. An extra column is removed. Put first the file that has the columns that you want to keep.
- Rows with an empty plot ID (`var`) stay in the output. These rows come from grid polygons that have no plot ID. You can remove them after the merge if necessary.

### 3. Split into a site workbook

`Supplemental/3_split_by_site_workbook.ipynb` puts the all-sites master CSV into one Excel workbook. Use it to share the data with people who work in Excel. This step is optional.

| Setting | Value |
|---|---|
| `input_file` | The all-sites master CSV |
| `output_file` | The path of the workbook to create |
| `sites` | Optional. A list of sites for the site sheets. `None` uses every site in the `Site` column. |

Output: a workbook with a `Combined` sheet (all rows) and one sheet for each site.

- The `Date` column is written as an Excel date (`mm/dd/yy`). If the notebook cannot read a date, the cell is empty.
- The notebook removes `Date_original` and similar columns if they exist.

## Method

For each plot polygon, all notebooks do these steps:

1. Cut the raster to the polygon with `rasterio.mask(..., crop=True, filled=True, nodata=0)`.
2. Calculate the mean of each band with `np.nanmean`.
3. Calculate each index from the band means.

The indices come from the plot-mean band values. They are not the mean of index values calculated for each pixel. For non-linear indices, these two methods can give different results.

**Known limitation:** with `filled=True, nodata=0`, pixels inside the plot bounding box but outside the polygon get the value 0. The mean includes these zeros. For plots that align with the raster grid, the effect is small. For rotated plots, the band means are lower than the true plot mean. All notebooks use the same method, so their outputs agree with each other.

### Differences between the template and batch notebooks

The batch notebooks use the same band order, masking, and formulas as the template notebooks. They add the code that is necessary to process many dates safely.

| | Template | Batch |
|---|---|---|
| Input | one raster path | one site folder, all dates |
| Plot ID | first column | first match from a list of column names |
| CRS | must be the same | shapefile changed to the raster CRS |
| Error for one plot | the run stops | `NaN` for that plot, the run continues |
| Division by zero (RGB) | `inf` or an error | `NaN` |

## Acknowledgements

The template notebooks were provided by Boubacar Gano, and the batch and supplemental notebooks were developed by Sasi Velchuri. Both in the Shakoor Lab, Donald Danforth Plant Science Center.
