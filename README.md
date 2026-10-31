# CardioMap Pro 2.0: Analytical Software for Cardiac Optical Mapping

A Streamlit application for exploring cardiac optical-mapping data. It loads voltage (Vm) and calcium (Ca) activation, duration and SNR maps stored as MATLAB `.mat` files. It then runs common electrophysiology analyses and adds deep-learning saliency maps.

## Analyses in the app (`elecMap.py`)

| Analysis | What it shows |
| --- | --- |
| Activation map | Thresholded activation pattern across the field of view |
| Conduction velocity | Velocity estimates derived from activation times |
| Action potential duration (APD) and repolarisation | Duration and repolarisation measures per region |
| Diastolic interval | Interval between successive beats |
| Signal processing and automatic segmentation | Filtering and segmentation of the optical signal |
| Region selection | Restrict any analysis to a chosen rectangle |
| Histogram and accuracy display | Value distributions, plus comparison with ground-truth maps |
| Grad-CAM | ResNet-18 saliency maps over the image |

`code for mat image.txt` holds an earlier variant that loads `.mat` files from a URL.

## Data in this repository

- `mat_api/`: example optical-mapping maps from pig preparations (2-week and 6-week; activation, duration, coupling and SNR) and rat preparations, plus processed integration maps.
- `mat_api/LICENSE.TXT`: MIT licence (© 2019 E. Branlard) for the bundled MAT-file helper code.
- `Train-/` and the `original_2D_01_*.jpg` images: cardiac MRI slices from the [Roboflow heart-segmentation-2 dataset](https://universe.roboflow.com/memoire/heart-segmentation-2/dataset/1) (CC BY 4.0), used for the image views.

## Running locally

```bash
pip install -r requirements.txt
streamlit run elecMap.py
```

## Related repositories

- [Advanced-Cardiac-Electrophysiology-Mapping-](https://github.com/datascintist-abusufian/Advanced-Cardiac-Electrophysiology-Mapping-): automated signal windowing and parallel processing for electrophysiology mapping

## Licence

No licence has been declared yet for the app code. The bundled `mat_api` helper code keeps its own MIT licence.

## Author

Md Abu Sufian - ORCID [0009-0007-3503-6942](https://orcid.org/0009-0007-3503-6942)
