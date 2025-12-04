# PGE_383_TermProject_XihanYao

**Exploring the Relationship Between Optimal K-Nearest Neighbors and Sample Density in a Geospatial Context**

## Project Overview

This project investigates how **sample density** influences the optimal value of *K* in K-Nearest Neighbors (KNN) regression models for geospatial machine learning. Using Land Surface Temperature (LST) data and multispectral satellite imagery from Austin, Texas, we demonstrate that optimal *K* is not solely determined by sampling density—it is fundamentally shaped by **feature selection** and the **informativeness of predictors**.

### Key Findings

- **Spatial coordinates only (x, y)**: Optimal *K* **decreases** as sample size increases (20 → 9)
- **Spatial coordinates + NDVI**: Optimal *K* **increases** as sample size increases (9 → 23)
- **Feature selection matters**: Including a strong linear predictor (NDVI, correlation ≈ −0.71 with temperature) reverses the relationship between sample size and optimal *K*

## Project Structure

```
PGE_383_TermProject_XihanYao/
├── README.md                                          # This file
├── LICENSE                                            # MIT License
├── Sample_Prep.ipynb                                  # Data preparation workflow
├── KNN_OptimalK_SampleDensity_Xihan_Yao.ipynb        # Main analysis notebook
├── data/                                              # Raw Landsat imagery (not included)
│   ├── Land_surface_temperature.tif
│   ├── Austin_B2_2025Aug.tif through Austin_B9_2025Aug.tif
│   └── ...
└── knn_experiment_results_*.csv                       # Results from large-scale experiments
```

## Workflow Overview

### Step 0: Data Preparation ([Sample_Prep.ipynb](Sample_Prep.ipynb))

Generates synthetic pseudo-samples from Landsat 8 imagery:

1. **Load LST data** from thermal infrared band
2. **Random sampling**: Draw 2,000 completely random points
3. **Hotspot-biased sampling**: Draw ~2,000 additional points near temperature hotspots (mimicking Urban Heat Island effect)
4. **Extract multispectral bands**: Extract reflectance values from Landsat bands B2–B9 at each sample location
5. **Calculate indices**: Compute NDVI, EVI, and NDWI for vegetation and water analysis
6. **Final dataset**: 3,924 pseudo-samples with bands, indices, and temperature labels

**Output**: LST_samples_with_bands_and_indices.csv

### Step 1–4: KNN Analysis ([KNN_OptimalK_SampleDensity_Xihan_Yao.ipynb](KNN_OptimalK_SampleDensity_Xihan_Yao.ipynb))

**Step 1**: Build a single KNN model and find optimal *K* (baseline)

**Step 2**: Examine the distribution of optimal *K* for a fixed sample size (1,000 iterations)

**Step 3**: Vary sample size (100–2,700) using only **x_coor, y_coor** as predictors
- Result: Optimal *K* decreases with sample size
- Interpretation: With sparse data, neighborhoods must be broad; with dense data, local neighbors suffice

**Step 4**: Repeat Step 3 using **x_coor, y_coor, NDVI** as predictors
- Result: Optimal *K* increases with sample size
- Interpretation: Strong linear predictors enable larger, more stable neighborhoods

## Dataset Description

### Study Area
- **Location**: Austin, Texas and surrounding areas
- **Bounding Box**: [−98.05, 30.10, −97.50, 30.55] (lon/lat)
- **Estimated Area**: ~55 km²
- **Imagery Source**: Landsat 8, Summer 2025 (publicly available via USGS EarthExplorer, Google Earth Engine, NASA LP DAAC)

### Sampling Strategy
- **Total Samples**: 3,924 pseudo-samples
  - 2,000 from random uniform sampling
  - ~1,924 from temperature-weighted hotspot sampling
- **Rationale**: Models realistic field campaign bias toward urban/warmer areas
- **No water body samples**: Valid; water exhibits different thermal and spectral properties

### Target Variable
- **Temperature (Land Surface Temperature, LST)**: Surface temperature derived from thermal infrared radiation (K)

### Predictor Variables Used in Analysis
| Variable | Description | Correlation with LST |
|----------|-------------|----------------------|
| `x_coor` | Column index (pixel position) | ≈ 0 |
| `y_coor` | Row index (pixel position) | ≈ 0 |
| `NDVI` | Normalized Difference Vegetation Index = $(B5 − B4)/(B5 + B4)$ | ≈ −0.71 |

### Additional Bands & Indices (Available)
| Code | Name | Formula/Use |
|------|------|-------------|
| B2 | Blue | Landsat band 2 |
| B3 | Green | Landsat band 3 |
| B4 | Red | Landsat band 4 |
| B5 | Near Infrared (NIR) | Landsat band 5 |
| B6 | SWIR1 | Shortwave infrared band 6 |
| B7 | SWIR2 | Shortwave infrared band 7 (useful for burned areas) |
| B8 | Panchromatic | High-resolution panchromatic band |
| B9 | Cirrus | High-altitude thin cloud detection |
| EVI | Enhanced Vegetation Index | $2.5 \cdot \frac{B5 − B4}{B5 + 6B4 − 7.5B2 + 1}$ |
| NDWI | Normalized Difference Water Index | $(B3 − B5)/(B3 + B5)$ |

## Key Results

### Experiment 1: X, Y Coordinates Only

```
Sample Size  | Avg Optimal K | Std Dev | Avg MSE
100          | 20.2          | 3.5     | 15.8
500          | 16.4          | 2.8     | 10.2
1000         | 13.7          | 2.4     | 8.9
2000         | 10.5          | 2.0     | 6.5
2700         | 9.1           | 1.8     | 5.2
```

**Trend**: $K_{opt} \downarrow$ as sample size $\uparrow$ → **Localization effect**

### Experiment 2: X, Y, NDVI

```
Sample Size  | Avg Optimal K | Std Dev | Avg MSE
100          | 9.4           | 2.1     | 8.2
500          | 12.8          | 2.4     | 3.5
1000         | 16.1          | 2.5     | 2.9
2000         | 19.7          | 2.6     | 1.8
2700         | 22.8          | 2.8     | 1.4
```

**Trend**: $K_{opt} \uparrow$ as sample size $\uparrow$ → **Smoothing effect**

## Technical Details

### Methods
- **Algorithm**: K-Nearest Neighbors Regression
- **Distance Metric**: Euclidean ($p = 2$)
- **Weight Function**: Uniform (all neighbors weighted equally)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Search Range**: $K \in [1, 50]$
- **Preprocessing**: StandardScaler (zero mean, unit variance)
- **Iterations per sample size**: 1,000 runs (to capture distribution variability)

### Statistical Approach
1. Treat full 3,924-sample dataset as "population"
2. For each sample size, draw 1,000 random subsamples (without replacement)
3. For each subsample, find optimal *K* via GridSearchCV
4. Record mean and standard deviation of optimal *K*
5. Plot trend across sample sizes

### Computational Cost
- ~27 sample sizes × 1,000 iterations × 50 *K* values ≈ **1.35 million KNN models**
- Runtime: Several hours on standard machine

## Dependencies

```python
numpy                 # Numerical computing
pandas                # Data manipulation
matplotlib            # Visualization
seaborn               # Statistical graphics
scikit-learn          # Machine learning (KNN, GridSearchCV, preprocessing)
rasterio              # Geospatial raster I/O
PIL                   # Image processing
```

Install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn rasterio pillow
```

## How to Reproduce

1. **Download/Access Data**
   - Use `Sample_Prep.ipynb` to generate pseudo-samples from Landsat imagery
   - Or download pre-generated samples: `LST_samples_with_bands_and_indices.csv`

2. **Run Analysis**
   - Open `KNN_OptimalK_SampleDensity_Xihan_Yao.ipynb` in Jupyter
   - Steps 1–2 run quickly (< 10 min)
   - Steps 3–4 require patience (2–3 hours) or load pre-computed results

3. **View Results**
   - Pre-computed results:
     - `knn_experiment_results_x_y.csv`
     - `knn_experiment_results_x_y_ndvi.csv`

## Implications & Discussion

This work demonstrates that **optimal *K* in KNN is not a fixed function of sampling density alone**. Instead, it emerges from the interplay of:

1. **Data geometry**: Spatial distribution of samples
2. **Feature informativeness**: Correlation and linearity of predictors
3. **Sample size**: Total available training data

### Practical Recommendations

- **For sparse, weakly-correlated features**: Use smaller *K* → favor local neighborhoods
- **For dense data with strong linear predictors**: Larger *K* may be optimal → favor broader averaging
- **Always tune *K***: Cross-validation hyperparameter search is essential; do not rely on rules of thumb alone
- **Consider feature engineering**: Adding informative features (like NDVI) fundamentally changes the optimal model structure

## Author

**Xihan Yao**
- PhD Student, Department of Geography and the Environment, UT Austin
- Research Focus: GIS, remote sensing, GeoAI, environmental modeling
- [LinkedIn](https://www.linkedin.com/in/xihan-yao-6381b3181/)

## Acknowledgments

- **Instructor**: Prof. Michael Pyrcz, Ph.D., P.Eng., Cockrell School of Engineering & Jackson School of Geosciences, UT Austin
  - [Website](http://michaelpyrcz.com) | [GitHub](https://github.com/GeostatsGuy) | [YouTube](https://www.youtube.com/channel/UCLqEr-xV-ceHdXXXrTId5ig)
- **Course TA**: Maria Gonzalez, Graduate Student, UT Austin
- Code and workflows adapted from Prof. Pyrcz's [MachineLearning Demos](https://github.com/GeostatsGuy/MachineLearningDemos_Book)

## References

Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is 'nearest neighbor' meaningful?. *Database Theory—ICDT 1999*, 217–235.

Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996). Geographically weighted regression: A method for exploring spatial nonstationarity. *Geographical Analysis*, 28(4), 281–298.

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.

Samworth, R. J. (2012). Optimal weighted nearest neighbour classifiers. *The Annals of Statistics*, 40(5), 2733–2763.

Weng, Q. (2009). Thermal infrared remote sensing for urban climate and environmental studies. *ISPRS Journal of Photogrammetry and Remote Sensing*, 64(4), 335–344.

## License

MIT License — See LICENSE file for details.

---

**Note on Data**: Large raster files (Landsat bands) are not included in this repository due to size constraints. These are publicly available through USGS EarthExplorer, Google Earth Engine, or NASA LP DAAC. The workflow demonstrates how to process and sample from such imagery.
