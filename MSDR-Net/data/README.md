# Data index format

The imaging data of the study cohort are not publicly available (see the main
README). To run the pipeline on your own data, prepare a CSV index with one row
per patient (one image per patient) and the following columns:

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier. All partitioning is performed at the patient level. |
| `image_path` | Path to the image, relative to `image_root` (or absolute). |
| `label` | 0 = benign, 1 = malignant (malignant is the positive class). |
| `split` | `train`, `val`, or `test` (patient-level 7:2:1 partition). |
| `center` | Center / institution identifier (used by `external_validation.py`). |
| `fold` | Integer 1–5, patient-level five-fold assignment on the combined train+val cohort (used by `cross_validation.py`). |
| `seed_x`, `seed_y` | Radiologist-placed seed point (pixel coordinates in the source image) used by `roi_crop.py`. |

Optional columns used by the rule-based cropping script, if a fixed initial
crop per case is preferred: none — the procedure is fully parameterized by
`roi_crop.py` arguments.

Example:

```csv
patient_id,image_path,label,split,center,fold,seed_x,seed_y
P0001,images/P0001.png,0,train,A,3,512,640
P0002,images/P0002.png,1,test,B,,388,517
```
