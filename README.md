# Yam Drone Phenotyping — Plot Indices and Canopy Cover

Tools from the Shakoor Lab (Donald Danforth Plant Science Center) for measuring yam field trials with drone imagery. The repo has two parts:

| Folder | What it does | Output |
|---|---|---|
| [`Indice-Extraction/`](Indice-Extraction/README.md) | Jupyter notebooks that calculate plot-level vegetation indices (25 RGB, 44 multispectral) from drone orthomosaics and a plot grid shapefile | One CSV per site/date, combined into site and all-site tables |
| [`Modeling/`](Modeling/README.md) | Python scripts that estimate per-plot canopy area with a U-Net canopy model and a seeded watershed, plus the trained 104-plot model and its training data | Per-plot canopy area (m²) for each site/date |

Both parts use the same inputs: an RGB (or multispectral) orthomosaic and the GRID shapefile for each site and date. Each part runs on its own. See the README in each folder for setup and run steps.

```
plot-indice-extraction/
├── Indice-Extraction/
│   ├── Original_Template_Script/   one image → one CSV (learn the method)
│   ├── VI_Batch/                   all dates for one site → one CSV per date
│   └── Supplemental/               merge date CSVs into site / all-site tables and an Excel workbook
└── Modeling/
    ├── Train/                      add annotations and retrain the canopy model
    ├── Weight/                     the current trained model
    ├── Test/                       run the model on selected site/dates
    ├── Full_canopy_Estimate/       run the model on every site/date
    └── Clipping/                   field-map helper for Godet25-26
```

Large data (orthomosaics, shapefiles, master workbooks) is not stored in this repo. It is on the Shakoor Lab OneDrive.

## Acknowledgements

A collaborative project between the Shakoor Lab (Donald Danforth Plant Science Center) and the DEFI Team (CIRAD).
