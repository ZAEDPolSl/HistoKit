# for single run
python run_segmentation_cohort.py --config ./configs/cohort.yaml

# with grid search
cd scripts_cohort
python run_segmentation_cohort.py --config ./configs/cohort_wsi.yaml --grid-config ./configs/grandqc_grid.yaml