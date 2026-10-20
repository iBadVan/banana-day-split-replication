# Banana leaf disease classification with a day-based test split

Replication package for the comparison of six ImageNet-pretrained models (ResNet50,
EfficientNetB0, MobileNetV2, ViT-B/16, Swin-T and CoAtNet-0) on the classification of
Sigatoka, Cordana and healthy banana leaves.

## Data

Banana Leaf Image Dataset for Classification of Sigatoka, Cordana, and Healthy Leaves,
Mendeley Data, version 3, DOI: 10.17632/hzttvh774g.3 (CC BY 4.0). 2,354 images in three
folders: `sigatoka` (995), `cordana` (815) and `sanas` (544, healthy).

## Split used in the paper

The test set contains four complete sampling days that never appear in training or
validation: 2025-01-08, 2025-01-14, 2025-02-09 and 2025-02-15.

| Class    | Train | Validation | Test | Total |
|----------|------:|-----------:|-----:|------:|
| Sigatoka |   796 |        100 |   99 |   995 |
| Cordana  |   649 |         82 |   84 |   815 |
| Healthy  |   441 |         54 |   49 |   544 |
| Total    | 1,886 |        236 |  232 | 2,354 |

`data/meta.csv` stores the subset of every image (`split`), its group (`group_id`) and its
capture day (`day`). Its rows follow the order of the image cache `X224.npy`.
`data/stats.json` stores the class weights computed on the training subset.
`data/grupos_enlaces.csv` lists the burst and visual-similarity links used to build the groups.

## Notebooks (run in this order)

| Notebook | Hardware | What it does |
|---|---|---|
| `00_banana_cache_224.ipynb` | CPU | Reads the dataset, corrects the EXIF orientation and resizes each full image to 224 x 224 (`X224.npy`). Its random 80/10/10 split is only an intermediate step. |
| `00b_reparticion_por_grupos.ipynb` | CPU, Internet | Data leakage audit: capture time from file names (bursts of 20 s or less), ResNet50 embeddings (cosine 0.92 or more) and SIFT matches (20 inliers or more). Writes `grupos_enlaces.csv`. |
| `00c_particion_por_dias.ipynb` | CPU | Final split: test by complete sampling days, validation by groups of 60 s or less. Writes `meta.csv` and `stats.json`. |
| `01_entrenamiento_comun.ipynb` | GPU | Trains one model (`MODEL`) with the common conditions and evaluates it on the test set. Run it once per model with `SMOKE_TEST = False`. |
| `99_comparacion_final.ipynb` | CPU | Joins the six results: global table, Wilson 95% intervals, McNemar test for the 15 pairs (with Bonferroni) and figures. |

To reproduce only the training, use the provided `data/meta.csv` and `data/stats.json` next to
the `X224.npy` produced by notebook 00, and run notebooks 01 and 99.

## Common training conditions

ImageNet-pretrained weights from timm, input 224 x 224, ImageNet normalization, no data
augmentation, AdamW (learning rate 1e-4, weight decay 1e-4), batch size 16, at most 50 epochs,
early stopping with patience 8 on the validation loss, ReduceLROnPlateau (factor 0.5,
patience 3), cross-entropy with class weights from the training subset, mixed precision and
random seed 42.

## Outputs of notebook 01 (per model)

`metrics_{MODEL}.json`, `history_{MODEL}.csv`, `pred_test_{MODEL}.csv`,
`fig_curves_{MODEL}.pdf` and `fig_cm_{MODEL}.pdf`.

## Methodology diagram

`figures/fig_methodology.pdf` (source: `figures/methodology.dot`, Graphviz).
