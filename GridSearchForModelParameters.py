# Import packages
import pandas as pd

# Create Dataframe ...........................................................

# Read in non-refugee and refugee centroid distances files
features = pd.read_csv('non_refugee_centroid_distances.csv', sep='\t')
features2 = pd.read_csv('refugee_centroid_distances.csv', sep='\t')

# Read in patient age file
age = pd.read_csv('ages_and_status.csv')

# Read in language ranking file
lang_rank = pd.read_csv('language_rank', sep='\t')

# Merge dataframes
del features2['Refugee_status']
features = features.merge(features2, how='left', on='Mother_MRN')
features['Age'] = age['Age'].values
features['language_rank'] = lang_rank['language_ranking'].values
features = features[(features['Refugee_status'] == 'yes') | (features['Refugee_status'] == 'no')].copy()

# Delete patient ID numbers
del features['Mother_MRN']

# Display all columns
pd.set_option('display.max_columns', None)

# Identify non-refugees as 0 and refugees as 1
features['Refugee_status'] = [0 if x == 'no' else 1 for x in features['Refugee_status']]

# Trai / Test / Split ........................................................

# Import packages
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# Create copy of features dataframe
df = features.copy()

# Display imbalance, need for SMOTE
ax = df['Refugee_status'].value_counts().plot(kind='bar', figsize=(10,6))

# Create X, y, training, testing sets
from sklearn.model_selection import train_test_split
X = df.drop('Refugee_status', axis=1)
y = df.Refugee_status
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Balance Data ...............................................................

# Import packages
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Create training and testing sets
XX, X_heldout, yy, y_heldout = train_test_split(X, y, test_size=0.3, random_state=None)

# Identify range of hyperparameters
n_estimators_grid = np.arange(80, 210, 20)
min_samples_split_grid = np.arange(2, 16, 2)
max_features_grid = np.arange(20, 26, 1)
min_samples_leaf_grid = np.arange(25, 55, 5)

# Balance training data with SMOTE
sm = SMOTE(random_state=None)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

# Display difference after SMOTE
print('Shape of X before SMOTE:', X.shape)
print('Shape of X after SMOTE:', X_sm.shape)
print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100

# Grid Search for Optimal Hyperparameters
mm_all = []
for n_estimators in n_estimators_grid:
    for min_samples_split in min_samples_split_grid:
        for max_features in max_features_grid:
            for min_samples_leaf in min_samples_leaf_grid:
                model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, class_weight='balanced_subsample', 
                                           criterion='entropy', max_features=max_features, min_samples_leaf=min_samples_leaf, random_state=None)
                cm_all = []
                k = 3
                
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
                cm_all.extend(cm)

                # Test fold 2
                X_sm, y_sm = sm.fit_resample(X_train_2, y_train_2)
                model.fit(X_sm, y_sm)
                preds = model.predict(X_fold_2)
                cm = confusion_matrix(y_fold_2, preds)
                cm_all.extend(cm) 

                # Test fold 1
                X_sm, y_sm = sm.fit_resample(X_train_3, y_train_3)
                model.fit(X_sm, y_sm)
                preds = model.predict(X_fold_1)
                cm = confusion_matrix(y_fold_1, preds)
                cm_all.extend(cm)
                
                tn = tp = fn = fp = 0;
                for i in np.array(range(0, len(cm_all), 2)):
                    tn = tn + cm_all[i][0]
                    fp = fp + cm_all[i][1]
                for i in np.array(range(1, len(cm_all), 2)):
                    fn = fn + cm_all[i][0]
                    tp = tp + cm_all[i][1]    
                mm  = [n_estimators, min_samples_split, max_features, min_samples_leaf, tp, tn, fp, fn]
                print(mm)
                mm_all.append(mm)
print('done')

# Create dataframe of performance metrics
perf_df = pd.DataFrame(mm_all, columns=['n_estimators', 'min_samples_split', 'max_features', 'min_samples_leaf', 'tp', 'tn', 'fp', 'fn'])

perf_df['total_neg'] = perf_df['tn'] + perf_df['fp']
perf_df['total_pos'] = perf_df['tp'] + perf_df['fn']
perf_df['acc'] = (perf_df['tp'] + perf_df['tn']) / (perf_df['total_neg'] + perf_df['total_pos'])
perf_df['precision'] = perf_df['tp'] / (perf_df['tp'] + perf_df['fp'])
perf_df['recall'] = perf_df['tp'] / (perf_df['tp'] + perf_df['fn'])
perf_df['specificity'] = perf_df['tn'] / perf_df['total_neg']

# Max accuracy
perf_df[perf_df.acc == perf_df.max()['acc']]

# Max specificity
perf_df[perf_df.specificity == perf_df.max()['specificity']]

# Max recall
perf_df[perf_df.recall == perf_df.max()['recall']]

# Add new column 'min_fn_fp': minimum combined fn + fp
perf_df['min_fn_fp'] = perf_df.fn + perf_df.fp

# # Find 'min_fp_fp' under 15
perf_df[perf_df.min_fn_fp < 15]

