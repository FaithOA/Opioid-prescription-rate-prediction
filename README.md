# Opioid-prescription-rate-prediction
Python - codeforprojectpaper.py
line 1 - 27 - Importing required packages for the analysis
line 28 - 30 - Reading in the joined data (due to data privacy and IRB, the data is not available for download)
line 31 - 37 - Creating region dictionary and region and GMEregion columns
line 38 - 82 - Creating categories for undergraduate medical education (UME) country, year of graduation from UME, year of graduation from GME
line 83 - 90 - Trimming the top and bottom 10% to remove outliers based on total claims
line 91 - 104 - Working with missing data and designating US born column as string to factorize the USBORN column to 0 - Not US-born, 1 - US-born, 2- Place of birth unknown
line 105 - 110 - Selecting features for the models
line 111 - 122 - Creating the SCORE column that divides prescribers into high-rate and low-rate opioid prescribers using total opioid claim rate
line 123 - 141 - Quality control to ensure the data split is correct
line 142 - 163 - One-hot encoding categorical variables, getting rid of duplicates
line 164 - 168 - Separating data into data and label using SCORE column as label
line 169 - 305 - Data processing, train/test split, Logistic regression model, Confusion matrix for LR model, ROC_AUC curve for LR model, Prediction-Recall curve for LR model
lines 306 - 463 - XGBoost model, Confusion matrix, ROC_AUC curve for both LR and XGBoost models in one plot, Prediction-Recall curve for LR and XGBoost models in one plot, feature importance plot for XGBoost.
lines 464 - 506 (Appendix code - takes a long time to run) - GridSearch, recursive feature elimination

R code - projectinr.R
Used for Odds ratios
lines 1 - 6 - Importing required packages for the analysis
line 7 - Importing the finalized data from python code
line 8 - 17 - Train/Test split
line 18 - 21 - logistic regression on train data
line 22 - 39 - McFadden and unpaired two-sample Wilcoxon test on high-rate (SCORE = 1) and lower-rate (SCORE = 0) prescribers
line 40 - 44 - Variable importance, odds ratios, and confidence intervals of regression model
