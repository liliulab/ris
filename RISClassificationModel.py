# Import packages
import pandas as pd

# Create Dataframe ...........................................................

# Read in non-refugee and refugee centroid distances files
features = pd.read_csv('non_refugee_centroid_distances.csv', sep='\t')
features2 = pd.read_csv('refugee_centroid_distances.csv', sep='\t')

# Read in age file, delete duplicate entries, delete patients who have not been identified
age = pd.read_csv('ages_and_status.csv')

# Read in language rank file
lang_rank = pd.read_csv('language_rank', sep='\t')

# Merge dataframes
del features2['Refugee_status']
features = features.merge(features2, how='left', on='Mother_MRN')
features['Age'] = age['Age'].values
features['language_rank'] = lang_rank['language_ranking'].values
features = features[(features['Refugee_status'] == 'yes') | (features['Refugee_status'] == 'no')].copy()

# Delete patient ID numbers
del features['Mother_MRN']

# Re-rank Spanish lanking ranking to 100
features['language_rank'] = features['language_rank'].replace([1], 100)

# Display all columns
pd.set_option('display.max_columns', None)

# Identify non-refugees as 0 and refugees as 1
features['Refugee_status'] = [0 if x == 'no' else 1 for x in features['Refugee_status']]
features

# Train / Test / Split .......................................................

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
import seaborn as sns

# Create copy of features dataframe
df = features.copy()

# Dislaly imbalance, need for SMOTE
ax = df['Refugee_status'].value_counts().plot(kind='bar', figsize=(10,6))

# Create X, y, training, testing sets
from sklearn.model_selection import train_test_split
X = df.drop('Refugee_status', axis=1)
y = df.Refugee_status
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# SMOTE Oversampling .........................................................s

# Import packages
from imblearn.over_sampling import SMOTE

# Balance training data
sm = SMOTE(random_state=None)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

# Display difference after SMOTE
print('Shape of X before SMOTE:', X.shape)
print('Shape of X after SMOTE:', X_sm.shape)
print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100

# Random Forest and Confusion Matrix .........................................

# Import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# Computer configuration (from grid search)
max_features_best = 21
min_samples_leaf_best = 40
min_samples_split_best = 14
n_estimators_best = 120

# Random forest model using optimal hyperparameters
model = RandomForestClassifier(n_estimators=n_estimators_best, min_samples_split=min_samples_split_best, 
                               class_weight='balanced_subsample', criterion='entropy', 
                               max_features=max_features_best, min_samples_leaf=min_samples_leaf_best, random_state=None)

# Fit random forest model
model.fit(X_sm, y_sm)
preds = model.predict(X_test)

# Print model accuracy and recall
print('Accuracy:', accuracy_score(y_test, preds))
print('Recall:', recall_score(y_test, preds))

# Display confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Performance results on held-out testing data
true_neg = true_pos = false_neg = false_pos = 0;
for i in np.array(range(0, len(cm), 2)):
    true_neg = true_neg + cm[i][0]
    false_pos = false_pos + cm[i][1]
for i in np.array(range(1, len(cm), 2)):
    false_neg = false_neg + cm[i][0]
    true_pos = true_pos + cm[i][1] 
    
# Calculate performance metrics
tot_neg = true_neg + false_pos
tot_pos = true_pos + false_neg
[tot_neg, tot_pos]
ac = (true_pos + true_neg)/(tot_neg + tot_pos)
prec = true_pos / (true_pos + false_pos)
re = true_pos/(true_pos + false_neg)
spec = true_neg/(true_neg + false_pos)
npv = true_neg / (true_neg + false_neg)

# Print results
print("Accuracy   :", ac)
print("Precision  :", prec)
print("Recall     :", re)
print("Specificity:", spec)
print("NPV        :", npv)
print("\ntp:", true_pos)
print("tn:", true_neg)
print("fp:", false_pos)
print("fn:", false_neg)

# Cross-Validation ...........................................................

# Create empty lists for cross-validation results
cm_all = []
fp_all_X = []
fn_all_X = []
fp_all = []
fn_all = []
prob_all = []

# Set k value (number of folds)
k=3
    
# Create training and testing folds
X_train, X_fold_1, y_train, y_fold_1 = train_test_split(X, y, test_size=1/k, random_state=None)
X_fold_2, X_fold_3, y_fold_2, y_fold_3 = train_test_split(X_train, y_train, test_size=0.5, random_state=None)

X_train_1 = X_fold_1.append(X_fold_2)
X_train_2 = X_fold_1.append(X_fold_3)
X_train_3 = X_fold_2.append(X_fold_3)

y_train_1 = y_fold_1.append(y_fold_2)
y_train_2 = y_fold_1.append(y_fold_3)
y_train_3 = y_fold_2.append(y_fold_3)

# Test fold 3
X_sm, y_sm = sm.fit_resample(X_train_1, y_train_1)
model.fit(X_sm, y_sm)
preds = model.predict(X_fold_3)
cm = confusion_matrix(y_fold_3, preds)

true_neg = true_pos = false_neg = false_pos = 0;
for i in np.array(range(0, len(cm), 2)):
    true_neg = true_neg + cm[i][0]
    false_pos = false_pos + cm[i][1]
for i in np.array(range(1, len(cm), 2)):
    false_neg = false_neg + cm[i][0]
    true_pos = true_pos + cm[i][1] 
    
print("True negative : ", true_neg)
print("False positive: ", false_pos)
print("False negative: ", false_neg)
print("True positive : ", true_pos)

cm_all.extend(cm)
    
# Find predicted probability
preds_prob = model.predict_proba(X_fold_3)
preds_prob = pd.DataFrame(preds_prob, columns=['prob_0', 'prob_1'])
preds_prob = pd.concat([pd.DataFrame(preds, columns=['pred']), preds_prob], axis=1)
preds_prob['prob'] = preds_prob[['prob_0', 'prob_1']].max(axis=1)
preds_prob['y'] = y_fold_3.values
preds_prob.index = y_fold_3.index
    
# Compare truth to predicted
prob_all.append(preds_prob)
comp = preds_prob
    
# Identify and display false negatives
fneg = comp[(comp.y == 1) & (comp.pred == 0)].index
fn_all_X.append(X.loc[fneg])
fn_all.append(comp.loc[fneg])
    
# Identify and display false positives
fpos = comp[(comp.y == 0) & (comp.pred == 1)].index
fp_all_X.append(X.loc[fpos])    
fp_all.append(comp.loc[fpos])  

# Test fold 2
X_sm, y_sm = sm.fit_resample(X_train_2, y_train_2)
model.fit(X_sm, y_sm)
preds = model.predict(X_fold_2)
cm = confusion_matrix(y_fold_2, preds)

true_neg = true_pos = false_neg = false_pos = 0;
for i in np.array(range(0, len(cm), 2)):
    true_neg = true_neg + cm[i][0]
    false_pos = false_pos + cm[i][1]
for i in np.array(range(1, len(cm), 2)):
    false_neg = false_neg + cm[i][0]
    true_pos = true_pos + cm[i][1] 
    
print("True negative : ", true_neg)
print("False positive: ", false_pos)
print("False negative: ", false_neg)
print("True positive : ", true_pos)

cm_all.extend(cm)
    
# Find predicted probability
preds_prob = model.predict_proba(X_fold_2)
preds_prob = pd.DataFrame(preds_prob, columns=['prob_0', 'prob_1'])
preds_prob = pd.concat([pd.DataFrame(preds, columns=['pred']), preds_prob], axis=1)
preds_prob['prob'] = preds_prob[['prob_0', 'prob_1']].max(axis=1)
preds_prob['y'] = y_fold_2.values
preds_prob.index = y_fold_2.index
    
# Compare truth to predicted
prob_all.append(preds_prob)
comp = preds_prob
    
# Identify and display false negatives
fneg = comp[(comp.y == 1) & (comp.pred == 0)].index
fn_all_X.append(X.loc[fneg])
fn_all.append(comp.loc[fneg])
    
# Identify and display false positives
fpos = comp[(comp.y == 0) & (comp.pred == 1)].index
fp_all_X.append(X.loc[fpos])    
fp_all.append(comp.loc[fpos])  

# Test fold 1
X_sm, y_sm = sm.fit_resample(X_train_3, y_train_3)
model.fit(X_sm, y_sm)
preds = model.predict(X_fold_1)
cm = confusion_matrix(y_fold_1, preds)

true_neg = true_pos = false_neg = false_pos = 0;
for i in np.array(range(0, len(cm), 2)):
    true_neg = true_neg + cm[i][0]
    false_pos = false_pos + cm[i][1]
for i in np.array(range(1, len(cm), 2)):
    false_neg = false_neg + cm[i][0]
    true_pos = true_pos + cm[i][1] 
    
print("True negative : ", true_neg)
print("False positive: ", false_pos)
print("False negative: ", false_neg)
print("True positive : ", true_pos)

cm_all.extend(cm)
    
# Find predicted probability
preds_prob = model.predict_proba(X_fold_1)
preds_prob = pd.DataFrame(preds_prob, columns=['prob_0', 'prob_1'])
preds_prob = pd.concat([pd.DataFrame(preds, columns=['pred']), preds_prob], axis=1)
preds_prob['prob'] = preds_prob[['prob_0', 'prob_1']].max(axis=1)
preds_prob['y'] = y_fold_1.values
preds_prob.index = y_fold_1.index
    
# Compare truth to predicted
prob_all.append(preds_prob)
comp = preds_prob
    
# Identify and display false negatives
fneg = comp[(comp.y == 1) & (comp.pred == 0)].index
fn_all_X.append(X.loc[fneg])
fn_all.append(comp.loc[fneg])
    
# Identify and display false positives
fpos = comp[(comp.y == 0) & (comp.pred == 1)].index
fp_all_X.append(X.loc[fpos])    
fp_all.append(comp.loc[fpos])  
    
fn_all_X = pd.concat(fn_all_X)
fp_all_X = pd.concat(fp_all_X)
prob_all = pd.concat(prob_all)

print('done')

# Cross-Validation Performance Results .......................................

# Calculate true negatives and true positives
tn = tp = fn = fp = 0;
for i in np.array(range(0, len(cm_all), 2)):
    tn = tn + cm_all[i][0]
    fp = fp + cm_all[i][1]
for i in np.array(range(1, len(cm_all), 2)):
    fn = fn + cm_all[i][0]
    tp = tp + cm_all[i][1]    

# Calculate accuracy, precision, recall, and specificity
total_neg = tn + fp
total_pos = tp + fn
[total_neg, total_pos]
acc = (tp + tn)/(total_neg + total_pos)
precision = tp / (tp + fp)
recall = tp/(tp + fn)
specificity = tn/(tn + fp)

# Print results
print("Accuracy   :", acc)
print("Precision  :", precision)
print("Recall     :", recall)
print("Specificity:", specificity)
print("\ntp:", tp)
print("tn:", tn)
print("fp:", fp)
print("fn:", fn)

# Cross-validation confusion matrix
cm_agg = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(10,7))
sns.heatmap(cm_agg, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# False Negatives ............................................................

print(fn_all_X.shape)
pd.concat([fn_all_X, pd.concat(fn_all)], axis=1, sort=True)

# False Positives ............................................................

print(fp_all_X.shape)
pd.concat([fp_all_X, pd.concat(fp_all)], axis=1, sort=True)

# ROC Curve ..................................................................

from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, _ = roc_curve(prob_all.y, prob_all.prob_1)
roc_auc = auc(fpr, tpr)
a = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
a.loc[abs(a.tpr - 0.85)<0.02, :]

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

# Plot a Decision Tree .......................................................

# Randomly sample 10 trees
import random
i = 0
trees = []
while (i < 10):
    t = random.randint(0, 119)
    trees.append(t)
    i += 1
print(trees)

# Decision tree
from sklearn.tree import export_graphviz, plot_tree
one_tree = model.estimators_[64]
plt.figure(figsize=(30,20))
plot_tree(one_tree, class_names=['0', '1'], feature_names=X.columns, fontsize=10, impurity=False, 
          filled=True, proportion=True)