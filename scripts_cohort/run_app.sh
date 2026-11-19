#!/bin/bash

python run_app.py \
  --csv "C:/Repos/HistoKit_current/HistoKit/Examples/Output/slide_stats.csv" \
  --thumbnails "C:/Repos/HistoKit_current/HistoKit/Examples/Output/tissue_segmentation/gamred/visualizations/thumbnails/" \
  --masks "C:/Repos/HistoKit_current/HistoKit/Examples/Output/artifact_segmentation/grandqc/visualizations/artifact_detection_map/" \
  --logo "C:/Repos/HistoKit_current/HistoKit/Examples/Output/artifact_segmentation/grandqc/visualizations/artifact_detection_map/C3N-02274-22.png"


python -m streamlit run C:\Repos\HistoKit_current\HistoKit\scripts_cohort\run_app.py