# Steel Surface Defect Analyzer

A Streamlit and OpenCV mini project implementing concepts from Foundations of Computer Vision Labs 1-5.

## Features

- Image reading and resizing
- Histogram equalisation
- Gaussian and median filtering
- Sobel and Canny edge detection
- Otsu, adaptive and K-means segmentation
- Morphological mask cleaning and defect-area estimation
- Harris, FAST, LBP and HOG feature extraction

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Import this repository into Streamlit Community Cloud and set the entrypoint to `app.py`.
