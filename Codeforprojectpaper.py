#Importing necessary packages
import pandas as pd
pd.set_option('display.max_rows', None, 'display.max_columns', None)
import numpy as np
from scipy.stats import zscore
from scipy import stats
from matplotlib import pyplot as plt
plt.rc('font', size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
from sklearn.metrics import confusion_matrix, f1_score
#reading in the joined deidentified data data
ops = pd.read_csv("/Users/Faith/Tables/ops.csv", encoding='latin-1')
ops.set_index("me", inplace=True)
#creating regions dictionary using state information
regions = {'IL':'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS':'Midwest', 'MI':'Midwest', 'MN':'Midwest', 'MO': 'Midwest', 'NE':'Midwest', 'ND': 'Midwest', 'OH':'Midwest', 'SD':'Midwest', 'WI': 'Midwest', 'CT':'Northeast', 'ME':'Northeast', 'MA': 'Northeast','NH':'Northeast', 'NJ':'Northeast', 'NY':'Northeast', 'PA':'Northeast', 'RI': 'Northeast', 'VT':'Northeast', 'AL': 'South', 'AR': 'South', 'DE' : 'South', 'DC':'South', 'FL': 'South', 'GA':'South','KY':'South','LA':'South', 'MD':'South', 'MS': 'South', 'NC': 'South', 'OK': 'South', 'SC':'South', 'TN':'South', 'TX': 'South', 'VA': 'South', 'WV' :'South', 'AK': 'West', 'AZ':'West', 'CA':'West', 'CO': 'West', 'HI' : 'West', 'ID':'West', 'MT':'West', 'NV':'West', 'NM':'West', 'OR':'West', 'UT':'West', 'WA': 'West', 'WY': 'West'}
print(regions)
#creating a region column using state information
ops['region'] = ops['state'].map(regions)
#creating a GME region column using gmestate information
ops['gmeregion'] = ops['gmestate'].map(regions)
print(ops.shape)
#designating UME country as either US or OTHER by creating a function
def ume_country(row):
    """ if the ume country row is US, the country is designated US, otherwise, designated as OTHER"""
    if row['umecountry'] == "US":
        return 'US'
    else:
        return 'OTHER'
ops.apply (lambda row: ume_country(row), axis=1)
ops['ume_country'] = ops.apply (lambda row: ume_country(row), axis=1)

#creating a function to categorize UME graduation period
def fixume_years(row):
    """ factorizing the UME graduation year into  5 categories"""
    if row['umegradyr'] <1990:
        return "Before 1990"
    elif row['umegradyr'] >=1990 and row['umegradyr'] <2000:
        return '1990 - 1999'
    elif row['umegradyr']>=2000 and row ['umegradyr'] <2010:
        return '2000 - 2009'
    elif row['umegradyr']>= 2010 and row ['umegradyr'] <2021:
        return '2010 and beyond'
    else:
        return 'NaN'

ops.apply(lambda row: fixume_years(row), axis=1)
ops['umegrad_group'] = ops.apply (lambda row: fixume_years(row), axis=1)

#creating a factor to categorize GME (residency) graduation period
def fixgme_years(row):
    """ factorizing the GME graduation year into  5 categories"""
    if row['gmegradyr'] < 1990:
        return "Before 1990"
    elif row['gmegradyr'] >=1990 and row['gmegradyr'] <2000:
        return '1990 - 1999'
    elif row['gmegradyr']>=2000 and row ['gmegradyr'] <2010:
        return '2000 - 2009'
    elif row['gmegradyr']>= 2010 and row ['gmegradyr'] <2021:
        return '2010 and beyond'
    else:
        return 'NaN'

ops.apply(lambda row: fixgme_years(row), axis=1)
ops['gmegrad_group'] = ops.apply (lambda row: fixgme_years(row), axis=1)
print(ops.head())
# trimming the top 10% and bottom 10% of the dataset using total claims.
s = ops.sort_values(by='tc', ascending=False)
trim1 = s.drop(s.index[range(17940)])
s2 = trim1.sort_values(by='tc')
trim = s2.drop(s2.index[range(17940)])
print(trim.shape)


#working with missing data, filling in with mean and creating a new value for the usborn missing.
mean_ageleavegme = trim['ageleavegme'].mean()
trim['ageleavegme'].fillna(value=mean_ageleavegme, inplace=True)
mean_gmegradyr = trim['gmegradyr'].mean()
trim['gmegradyr'].fillna(value=mean_gmegradyr, inplace=True)
mean_odpc = trim['odpc'].mean()
trim['odpc'].fillna(value=mean_odpc, inplace=True)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='constant', fill_value=2)
trim["usborn"] = imp.fit_transform(trim[["usborn"]]).ravel()

#designating the USborn column as string type (factor)
trim['usborn'] = trim['usborn'].astype(str)

#selecting the required columns for the models
selcols = ['ageleavegme', 'usborn', 'region', 'gdr',  'gmeregion', 'ume_country', 'umegrad_group', 'gmegrad_group', 'specgrp','tocr', 'odpc', 'region', 'gmeregion', 'tocr_cy']
selected =  trim.loc[:, selcols].copy()
selected = selected.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)
print(selected.shape)

#creating the score column that classifies top 10% as high-volume prescribers and the rest as lower-volume
selected["SCORE"] = pd.qcut(selected["tocr"], q=[0, 0.9, 1],
                            labels=[0,1 ])
print(selected['SCORE'].value_counts()[1])

selord = selected.sort_values(by='tocr', ascending=False)
selord['cum_tocr'] = selord['tocr'].cumsum()
selord['cum_percent'] = 100 * (selord['tocr'].cumsum()/selord['tocr'].sum())


len(selected[selected.SCORE == 1])
print(selected.shape)
#getting the top 10% of providers as high-volume prescribers
topten = selected.loc[(selected['SCORE'] ==1)]
print(topten.shape)
#getting the bottom 90%
bottom = selected.loc[(selected['SCORE'] ==0)]
print(bottom.shape)
#getting the shape of the NE data from the top 10%
NE =selected.loc[selected['usborn'] == 2]
print(NE.shape)

#makng sure that the percentages are correct for the designation as high-rate and low-rate
low_prescribers = len(selected[selected['SCORE'] == 0])
high_prescribers = len(selected[selected['SCORE'] == 1])
perc_low = low_prescribers/(low_prescribers+high_prescribers)
print('percentage of 0 is ', perc_low*100)
perc_high = high_prescribers/(low_prescribers+high_prescribers)
print('percentage of 1 is', perc_high*100)

#features for the model
Features = ["ageleavegme", "odpc", "gender", "usborn", "specialty group", "UME country", "current state", "GME state", "UME country", "UME graduation year", "GME graduation year", "tocr_cy"]
print('Features:', Features)


selected2cols = ['ageleavegme', 'usborn', 'gdr', 'ume_country', 'gmeregion', 'region', 'umegrad_group', 'gmegrad_group', 'specgrp', 'odpc', 'SCORE', 'tocr_cy']
selected2 = selected.loc[:, selected2cols].copy()
#one-hot encoding categorical variables
vars = ['gdr', 'usborn', 'specgrp', 'ume_country', 'region', 'gmeregion', 'umegrad_group', 'gmegrad_group']
for var in vars:
    cat_list='var'+ '_'+var
    cat_list = pd.get_dummies(trim[var], prefix=var)
    data1 = selected2.join(cat_list)
    selected2 = data1
#Getting rid of duplicates
vars = ['gdr','usborn', 'region', 'ume_country', 'gmeregion', 'specgrp','umegrad_group', 'gmegrad_group']
selected_var =selected2.columns.values.tolist()
to_keep=[i for i in selected_var if i not in vars]
print(to_keep)
selected_final =selected2[to_keep]

print(selected_final.columns.values)
print(selected_final.head())
#separating into data and labels using SCORE as label
X = selected_final.loc[:, selected_final.columns != 'SCORE']
y = selected_final.loc[:, selected_final.columns == 'SCORE']
y = y.to_numpy().ravel()

#using StandardScaler to scale the data for better use in logistic regression model
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#test-train split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)


#generating a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
#calling the regression model
logreg = LogisticRegression(solver='saga',max_iter=2500, class_weight={0: 0.1, 1: 0.90})
logreg.fit(X_train, y_train)

threshold = 0.478027
y_pred = (logreg.predict(X_test)> threshold).astype('float')

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('F1 score', f1_score(y_test, y_pred))

#creating confusion matrix for logistic regression
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

total1=sum(sum(confusion_matrix))
#calculating accuracy from confusion matrix
accuracy1=(confusion_matrix[0,0]+confusion_matrix[1,1])/total1
print ('Accuracy for logistic regression : ', accuracy1)
#calculating sensitivity from confusion matrix
sensitivity1 = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Sensitivity for LR: ', sensitivity1 )
#calculating specificity from confusion matrix
specificity1 = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Specificity for LR: ', specificity1)

#plotting confusion matrix in heatmap format
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', vmin=0, vmax=35000,  annot_kws={"fontsize":30})
ax.tick_params(labelsize = 15)
ax.set_title('Confusion Matrix for Logistic Regression\n\n', fontsize=22, fontweight = 'bold')
ax.set_xlabel('\nPredicted label', fontsize = 20, fontweight = "bold")
ax.set_ylabel('Actual label ', fontsize = 20, fontweight='bold')
ax.xaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], fontsize = 18, fontweight = 'bold')
ax.yaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], rotation = 'horizontal', fontsize = 18, fontweight='bold')

#Visualizing the confusion matrix
plt.show()

#calculating matthews coefficient
print('Matthews', matthews_corrcoef(y_test, y_pred))
#calculating ROC_AUC score
print('roc auc', roc_auc_score(y_test, y_pred))

#generating classification report
print(classification_report(y_test, y_pred))

#calculating precision score
print("Precision score: {}".format(precision_score(y_test,y_pred)))
print("Roc AUC: ", roc_auc_score(y_test, y_pred, average='macro'))
#calculating ROC_AUC Score
logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1])

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1] )
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
from numpy import sqrt
from numpy import argmax
# get the best threshold
J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
plt.figure()
colors = ['red', 'blue']
ax.set_prop_cycle('color', colors)
# plot the roc curve for the model
l1 = plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color = '#9E3C0E', linewidth = 3)
l2 = plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc, color = 'green', linewidth = 3)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=16, fontweight = 'bold')
plt.ylabel('True Positive Rate', fontsize= 16, fontweight = 'bold')
plt.grid(False)
plt.title('Logistic Regression Receiver Operating Characteristic', fontsize = 18, fontweight = "bold")
legend_properties = {'weight':'bold'}
plt.legend(loc="lower right", prop=legend_properties)
plt.savefig('Log_ROC')

plt.show()

#plotting precision-recall curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, logreg.predict_proba(X_test)[:,1])
lr_f1, lr_auc = f1_score(y_test,y_pred ), auc(lr_recall, lr_precision)
#To summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y[y==1])/len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', linewidth =3)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression', linewidth=3)
#Calculating F1 score
fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
#locating the index of the largest F1 score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# axis labels
plt.title("Precision Recall Curve for Logistic Regression. F1 = 0.29, AUC = 0.25", fontsize = 18, fontweight = 'bold')
plt.xlabel('Recall', fontsize = 16, fontweight = 'bold')
plt.grid(False)
plt.ylabel('Precision', fontsize = 16, fontweight = 'bold')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(np.arange(0, 1.1, step=0.1), fontweight='bold')
plt.yticks(np.arange(0, 1.1, step=0.1), fontweight='bold')
# show the legend
plt.legend(prop=legend_properties)
# show the plot
plt.show()


#importing XGBoost
import xgboost as xgb
#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#calling the classifier
xgbs = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic', n_estimators = 350, min_child_weight = 1, max_depth = 4, learning_rate = 0.2, gamma = 5, colsample_bytree =1.0, scale_pos_weight=9, subsample =0.8)

xgbs.fit(X_train, y_train)
THRESHOLD = 0.264891
preds = np.where(xgbs.predict(X_test)> THRESHOLD, 1, 0)

#Accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" %(accuracy))
#Dataset Conversion to DMatrix
churn_dmatrix = xgb.DMatrix(data = X, label =  y)

#Specifying parameters
params = {"objective" : "binary:logistic", "max_depth" : 4}

#Performing Cross-validation
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, nfold = 4, num_boost_round = 10, metrics = "error", as_pandas = True)

#Accuracy of CV
print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))
print("Roc AUC: ", roc_auc_score(y_test, preds, average='macro'))
probs = xgbs.predict_proba(X_test)



#Matthews correlation coefficient
print('Matthews', matthews_corrcoef(y_test, preds))
from sklearn.metrics import average_precision_score
print (average_precision_score(y_test, preds))

from sklearn.metrics import confusion_matrix
confusion_matrix2 = confusion_matrix(y_test, preds)
print(confusion_matrix2)
total1=sum(sum(confusion_matrix2))
#####from confusion matrix calculate accuracy
accuracy1=(confusion_matrix2[0,0]+confusion_matrix2[1,1])/total1
print ('Accuracy for xgboost : ', accuracy1)

sensitivity1 = confusion_matrix2[0,0]/(confusion_matrix2[0,0]+confusion_matrix2[0,1])
print('Sensitivity for xgboost: ', sensitivity1 )

specificity1 = confusion_matrix2[1,1]/(confusion_matrix2[1,0]+confusion_matrix2[1,1])
print('Specificity for xgboost: ', specificity1)

print(classification_report(y_test,preds))
print('F1 score', f1_score(y_test, preds))
from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,preds)))
#displaying confusion matrix in heatmap format
ax = sns.heatmap(confusion_matrix2, annot=True, cmap='Blues', fmt='g', vmin=0, vmax=35000,  annot_kws={"fontsize":30})

ax.set_title('Confusion Matrix for XGBoost\n\n', fontsize=22, fontweight = 'bold')
ax.set_xlabel('\nPredicted label', fontsize = 20, fontweight = 'bold')
ax.set_ylabel('Actual label ', fontsize = 20, fontweight = 'bold')
ax.xaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], fontsize = 18, fontweight = 'bold')
ax.yaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], rotation = 'horizontal', fontsize = 18, fontweight = 'bold')
## Display the visualization of the Confusion Matrix.
plt.show()
#Best threshold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc2 = roc_auc_score(y_test, xgbs.predict_proba(X_test)[:,1])
fpr2, tpr2, thresholds = roc_curve(y_test, xgbs.predict_proba(X_test)[:,1])
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
from numpy import sqrt
from numpy import argmax
# get the best threshold
J = tpr2 - fpr2
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
#Plotting ROC_AUC curves for both logistic regression and XGboost models
plt.figure()
colors = ['red', 'blue']
ax.set_prop_cycle('color', colors)
# plot the roc curve for the model
l1 = plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill',  color = 'blue',linewidth = 3)
l2 = plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc, linewidth = 3)
l3 = plt.plot(fpr2, tpr2, label='XGBoost (area = %0.2f)' % logit_roc_auc2,  linewidth=3)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18, fontweight = 'bold')
plt.ylabel('True Positive Rate', fontsize= 18, fontweight = 'bold')
plt.grid(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.title('Receiver Operating Characteristic for Both Models', fontsize = 18, fontweight ='bold')
plt.legend(loc="lower right", prop=legend_properties)
plt.savefig('Log_ROC')
plt.xticks(fontweight='bold', fontsize = 14)
plt.yticks(fontweight='bold', fontsize = 14)
plt.show()

#precision-recall curves for both logistic regression and XGBoost models
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
lr_precision2, lr_recall2, _ = precision_recall_curve(y_test, xgbs.predict_proba(X_test)[:,1])
lr_f2, lr_auc2 = f1_score(y_test,y_pred ), auc(lr_recall2, lr_precision2)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f2, lr_auc2))
# plot the precision-recall curves
no_skill = len(y[y==1])/len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', linewidth=3)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression', linewidth=3)
plt.plot(lr_recall2, lr_precision2, marker='.', label='XGBoost', linewidth=3)
# axis labels
plt.title("Precision-Recall Curves for Both Models", fontsize = 18, fontweight='bold')
plt.xlabel('Recall', fontsize = 20, fontweight = 'bold')
plt.ylabel('Precision', fontsize = 20,fontweight = 'bold')
plt.grid(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xticks(np.arange(0, 1.1, step=0.1), fontweight='bold', fontsize = 14)
plt.yticks(np.arange(0, 1.1, step=0.1), fontweight='bold', fontsize = 14)
#Show legend
plt.legend(prop=legend_properties)
#Display plot
plt.show()

#creating the feature importance plot for XGBoost model
xgbs.get_booster().feature_names = ["Age at GME Graduation", "Opioid Days Per Claim", "County Total Opioid Claim Rate", "Female Gender",  "Male Gender", "Foreign Born", "US Born", "Place of Birth - Unknown", "Family Medicine", "Internal Medicine", "Non-US UME Graduate", "UME Country - US", "Current Region - Midwest", "Current Region - Northeast", "Current Region - South", "Current Region - West", "GME Region - Midwest", "GME Region - Northeast",  "GME Region - South", "GME Region - West", "UME Graduation 1990 - 1999", "UME Graduation 2000 - 2009", "UME Graduation After 2010","UME Graduation Before 1990", "GME Graduation 1990 - 1999", "GME Graduation 2000 - 2009", "GME Graduation After 2010", "GME Graduation Before 1990", "GME Graduation Unknown"]
xgb.plot_importance(xgbs.get_booster(), importance_type='weight', height = 1)
plt.title("Feature Importance From XGBoost Model", fontsize = 18, fontweight = 'bold')
plt.grid(False)
plt.xlabel('F Score', fontsize = 18, fontweight = 'bold')
plt.ylabel('Features', fontsize = 18, fontweight = 'bold')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(fontweight='bold', fontsize = 14)
plt.yticks(fontweight='bold', fontsize = 14)
plt.show()



#APPENDIX CODE
#getting the best class weights
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(solver='newton-cg')
#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)
#Grid search
#Dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr,
                          param_grid= param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1',
                          verbose=2).fit(X_train, y_train)

#Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)

#recursive feature elimination
selected_final_vars=selected.columns.values.tolist()
y= ['SCORE']
X=[i for i in trim_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X_scaled, y_train)
print(rfe.support_)
print(rfe.ranking_)

from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))




#XGBoost grid search
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5],
    'learning_rate' : [0.01, 0.1, 0.2],
    'n_estimators' : [200, 300, 400]
}
xgbs = xgb.XGBClassifier( objective='binary:logistic',
                    silent=True, nthread=1)
folds = 3
param_comb = 7

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgbs, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train, y_train)
timer(start_time) # timing ends here for "start_time" variable

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
import os
os.makedirs('/Users/Faith/Tables', exist_ok=True)
results.to_csv('/Users/Faith/Tables/gridsearch.csv', index=False)


#Sensitivity analysis
sens1 = pd.read_csv("/Users/Faith/Tables/top20percentile.csv", encoding='latin-1')
sens1.set_index("me", inplace=True)
sens2 = pd.read_csv("/Users/Faith/Tables/bottom20percentile.csv", encoding='latin-1')
sens2.set_index("me", inplace=True)
sensitivitydata = pd.concat([sens1, sens2])
sensitivitydata = sensitivitydata.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)

print(sens1.shape)
print(sens2.shape)
print(sensitivitydata.shape)
senss = sensitivitydata.sort_values(by='tocr', ascending=False)
senss["SCORE2"] = pd.qcut(senss["tocr"], q=[0, 0.5, 1],
                          labels=[0,1 ])

low_prescribers = len(selected[selected['SCORE'] == 0])
high_prescribers = len(selected[selected['SCORE'] == 1])
perc_low = low_prescribers/(low_prescribers+high_prescribers)
print('percentage of 0 is ', perc_low*100)
perc_high = high_prescribers/(low_prescribers+high_prescribers)
print('percentage of 1 is', perc_high*100)
Features = ["ageleavegme", "odpc", "gender", "usborn", "specialty group", "UME country", "current state", "GME state", "UME country", "UME graduation year", "GME graduation year", "tocr_cy"]
print('Features:', Features)

print(selected.describe())

selected2cols = ['ageleavegme', 'usborn', 'gdr', 'ume_country', 'gmeregion', 'region', 'umegrad_group', 'gmegrad_group', 'specgrp', 'odpc', 'SCORE2', 'tocr_cy']
selected2 = senss.loc[:, selected2cols].copy()
vars = ['gdr', 'usborn', 'specgrp', 'ume_country', 'region', 'gmeregion', 'umegrad_group', 'gmegrad_group']
for var in vars:
    cat_list='var'+ '_'+var
    cat_list = pd.get_dummies(trim[var], prefix=var)
    data1 = selected2.join(cat_list)
    selected2 = data1

vars = ['gdr','usborn', 'region', 'ume_country', 'gmeregion', 'specgrp','umegrad_group', 'gmegrad_group']
selected_var =selected2.columns.values.tolist()
to_keep=[i for i in selected_var if i not in vars]
print(to_keep)
selected_final =selected2[to_keep]

print(selected_final.columns.values)

X = selected_final.loc[:, selected_final.columns != 'SCORE2']
y = selected_final.loc[:, selected_final.columns == 'SCORE2']
y = y.to_numpy().ravel()
from sklearn.linear_model import LogisticRegression
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
logreg = LogisticRegression(solver='saga',max_iter=2500)
logreg.fit(X_train, y_train)

threshold = 0.42
y_pred = (logreg.predict(X_test)> threshold).astype('float')

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print('F1 score', f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix, f1_score


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', vmin=0, vmax=35000,  annot_kws={"fontsize":30})
ax.tick_params(labelsize = 15)
ax.set_title('Confusion Matrix for Logistic Regression\n\n', fontsize=22, fontweight = 'bold')
ax.set_xlabel('\nPredicted label', fontsize = 20, fontweight = "bold")
ax.set_ylabel('Actual label ', fontsize = 20, fontweight='bold')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], fontsize = 18, fontweight = 'bold')
ax.yaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], rotation = 'horizontal', fontsize = 18, fontweight='bold')

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import matthews_corrcoef
print('Matthews', matthews_corrcoef(y_test, y_pred))
from sklearn.metrics import roc_auc_score
print('roc auc', roc_auc_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,y_pred)))
print("Roc AUC: ", roc_auc_score(y_test, y_pred, average='macro'))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1])

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1] )
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
from numpy import sqrt
from numpy import argmax
# get the best threshold
J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
plt.figure()
colors = ['red', 'blue']
ax.set_prop_cycle('color', colors)
# plot the roc curve for the model
l1 = plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color = '#9E3C0E', linewidth = 3)
l2 = plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc, color = 'green', linewidth = 3)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=16, fontweight = 'bold')
plt.ylabel('True Positive Rate', fontsize= 16, fontweight = 'bold')
plt.grid(False)
plt.title('Logistic Regression Receiver Operating Characteristic', fontsize = 18, fontweight = "bold")
legend_properties = {'weight':'bold'}
plt.legend(loc="lower right", prop=legend_properties)
plt.savefig('Log_ROC')

plt.show()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
lr_precision, lr_recall, thresholds = precision_recall_curve(y_test, logreg.predict_proba(X_test)[:,1])
lr_f1, lr_auc = f1_score(y_test,y_pred ), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y[y==1])/len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', linewidth =3)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression', linewidth=3)
# convert to f score
fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

# axis labels
plt.title("Precision Recall Curve for Logistic Regression. F1 = 0.29, AUC = 0.25", fontsize = 18, fontweight = 'bold')
plt.xlabel('Recall', fontsize = 16, fontweight = 'bold')
plt.grid(False)
plt.ylabel('Precision', fontsize = 16, fontweight = 'bold')
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(np.arange(0, 1.1, step=0.1), fontweight='bold')
plt.yticks(np.arange(0, 1.1, step=0.1), fontweight='bold')
# show the legend
plt.legend(prop=legend_properties)
# show the plot
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
xgbs = xgb.XGBClassifier(booster='gbtree', objective='binary:logistic', n_estimators= 350, seed = 123, learning_rate=0.2, scale_pos_weight=1, max_depth=4)

xgbs.fit(X_train, y_train)
THRESHOLD = 0.50

preds = np.where(xgbs.predict(X_test)> THRESHOLD, 1, 0)

#Accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" %(accuracy))
#Dataset Conversion to DMatrix
churn_dmatrix = xgb.DMatrix(data = X, label =  y)

#Specifying parameters
params = {"objective" : "binary:logistic", "max_depth" : 4}

#Performing Cross-validation
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, nfold = 4, num_boost_round = 10, metrics = "error", as_pandas = True)


#Accuracy
print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1]))
print("Roc AUC: ", roc_auc_score(y_test, preds, average='macro'))
probs = xgbs.predict_proba(X_test)



from sklearn.metrics import matthews_corrcoef
print('Matthews', matthews_corrcoef(y_test, preds))
from sklearn.metrics import average_precision_score
print (average_precision_score(y_test, preds))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, preds)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test,preds))
print('F1 score', f1_score(y_test, preds))
from sklearn.metrics import precision_score

print("Precision score: {}".format(precision_score(y_test,preds)))

ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', vmin=0, vmax=35000,  annot_kws={"fontsize":30})

ax.set_title('Confusion Matrix for XGBoost\n\n', fontsize=22, fontweight = 'bold')
ax.set_xlabel('\nPredicted label', fontsize = 20, fontweight = 'bold')
ax.set_ylabel('Actual label ', fontsize = 20, fontweight = 'bold')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], fontsize = 18, fontweight = 'bold')
ax.yaxis.set_ticklabels(['Low-volume Prescriber: 0','High-volume Prescriber: 1'], rotation = 'horizontal', fontsize = 18, fontweight = 'bold')
## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc2 = roc_auc_score(y_test, xgbs.predict_proba(X_test)[:,1])
fpr2, tpr2, thresholds = roc_curve(y_test, xgbs.predict_proba(X_test)[:,1])
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
from numpy import sqrt
from numpy import argmax
# get the best threshold
J = tpr2 - fpr2
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
plt.figure()
colors = ['red', 'blue']
ax.set_prop_cycle('color', colors)
# plot the roc curve for the model
l1 = plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill',  color = 'blue',linewidth = 3)
l2 = plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc, linewidth = 3)
l3 = plt.plot(fpr2, tpr2, label='XGBoost (area = %0.2f)' % logit_roc_auc2,  linewidth=3)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18, fontweight = 'bold')
plt.ylabel('True Positive Rate', fontsize= 18, fontweight = 'bold')
plt.grid(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.title('Receiver Operating Characteristic for Both Models', fontsize = 18, fontweight ='bold')
plt.legend(loc="lower right", prop=legend_properties)
plt.savefig('Log_ROC')
plt.xticks(fontweight='bold', fontsize = 14)
plt.yticks(fontweight='bold', fontsize = 14)
plt.show()

threshold = 0.48


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
lr_precision2, lr_recall2, _ = precision_recall_curve(y_test, xgbs.predict_proba(X_test)[:,1])
lr_f2, lr_auc2 = f1_score(y_test,y_pred ), auc(lr_recall2, lr_precision2)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f2, lr_auc2))
# plot the precision-recall curves
no_skill = len(y[y==1])/len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', linewidth=3)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic Regression', linewidth=3)
plt.plot(lr_recall2, lr_precision2, marker='.', label='XGBoost', linewidth=3)


# axis labels
plt.title("Precision Recall Curves for Both Models", fontsize = 18, fontweight='bold')
plt.xlabel('Recall', fontsize = 20, fontweight = 'bold')
plt.ylabel('Precision', fontsize = 20,fontweight = 'bold')
plt.grid(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['top'].set_color('black')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['bottom'].set_color('black')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xticks(np.arange(0, 1.1, step=0.1), fontweight='bold', fontsize = 14)
plt.yticks(np.arange(0, 1.1, step=0.1), fontweight='bold', fontsize = 14)
# show the legend
plt.legend(prop=legend_properties)
# show the plot
plt.show()



