
#!pip install missingno
import pandas as pd
import math
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import missingno as mno
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

file_name = 'fr.openfoodfacts.org.products.csv'
target_local = 'fr.openfoodfacts.org.products.csv'
target_local_samples = 'fr.openfoodfacts.org.products.sample_1000.csv'
workplace = 'LOCAL'
#workplace = 'LOCAL_SAMPLE'
#workplace = 'GCLOUD'

pd.options.display.max_rows = 170
pd.set_option('display.multi_sparse', False)

#% matplotlib inline
sns.set_style()

if 'LOCAL' == workplace:
    food_data = pd.read_csv(target_local, sep='\t', low_memory=False, dtype={'code': 'object'})
    target = target_local
elif 'LOCAL_SAMPLE' == workplace:
    food_data = pd.read_csv(target_local_samples, low_memory=False, dtype={'code': 'object'})
    target = target_local_samples
else:
    print('Unknown workplace')
    

#Convert to date
food_data['created_datetime'] = pd.to_datetime(food_data['created_datetime'], errors='coerce', infer_datetime_format=True)
food_data['last_modified_datetime'] = pd.to_datetime(food_data['last_modified_datetime'], errors='coerce', infer_datetime_format=True)

#Missing data
food_data_count = food_data.count('index')
food_data_rows = len(food_data)
missing_data = pd.DataFrame({'Percentage': 1-(food_data_count/food_data_rows), 'Count': food_data_rows-food_data_count}).sort_values(by='Count')
food_data_clean = food_data.copy()
na_percentage_cutoff = 0.8 # 0.6 --> columns with more than 60% missing values will be removed
valid_cols = missing_data[missing_data['Percentage']<=na_percentage_cutoff].index.values
food_data_clean = food_data_clean[valid_cols]

#Let's remove the duplicated fields and keep only english tagging
for f in food_data_clean.columns.values:
    tags_n_fr = [f+'_tags', f+'_fr']
    food_data_clean = food_data_clean.drop(tags_n_fr, axis=1, errors='ignore')
    
# And remove additional columns that won't of any use
columns_not_used = ['url', 'created_t', 'last_modified_t', 'image_small_url', 'image_url', 'brands', 'ingredients_text', 'quantity', 'packaging', 'ingredients_that_may_be_from_palm_oil_n', 'ingredients_from_palm_oil_n', 'states', 'serving_size', 'categories']
food_data_clean = food_data_clean.drop(columns_not_used, axis=1, errors='ignore')
food_data_clean = food_data_clean.drop(index=food_data_clean[food_data_clean['product_name'].isna()].index.values)
food_data_clean = food_data_clean.drop(index=food_data_clean[food_data_clean['code'].isna()].index.values)
food_data_clean[['pnns_groups_1', 'pnns_groups_2']] = food_data_clean[['pnns_groups_1', 'pnns_groups_2']].replace('unknown', np.nan)

# Create reference table
food_ref_table = food_data_clean.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# Standardize the reference table (only numeric columns)
scaler = StandardScaler().fit(food_ref_table.select_dtypes(include=[np.number]))
numeric_columns = food_ref_table.select_dtypes(include=[np.number]).columns.tolist()
food_ref_table_n = food_ref_table.copy()
food_ref_table_n[numeric_columns] = scaler.transform(food_ref_table_n[numeric_columns])

#food_ref_table_n_train, food_ref_table_n_test = train_test_split(food_ref_table_n, train_size=0.7)

kfold = KFold(n_splits=5, shuffle=True)
K_max = 10
mse_total = {k: [] for k in food_ref_table_n[numeric_columns]}
for train, test in kfold.split(food_ref_table_n):
    food_ref_table_n_train = food_ref_table_n.iloc[train]
    food_ref_table_n_test = food_ref_table_n.iloc[test]

    for field in food_ref_table_n_test[numeric_columns]:
#        mse_total[field] = []
        # Using sklearn NearestNeighbors return n_neighbors for each row order by ascending distances
        neighbors = NearestNeighbors(n_neighbors=K_max, metric='euclidean').fit(food_ref_table_n_train[numeric_columns].drop(columns=(field)))
        indices = neighbors.kneighbors(food_ref_table_n_test[numeric_columns].drop(columns=(field)), return_distance=False)
        
        # Finding the K_max nearest neighbors
        neighbors_values = []
        for index in indices:
            neighbors_values.append(food_ref_table_n_train.iloc[index][field].values)       
        
        # Computing the mean for K neighbors with K=1...K_max and corresponding MSE
        mse = []
        for k in range(1, K_max):
            predicted = np.array(neighbors_values)[:,0:k].mean(axis=1)
            mse.append(mean_squared_error(food_ref_table_n_test[field], predicted))
    
        mse_total[field].append(np.array(mse))

mse_cv = {}    
field_optimal_K = {}
for field, val in mse_total.items():
    mse_cv[field] = np.array(val).mean(axis=0)
    print(field, "K with minimal MSE:", np.argmax(mse_cv[field])+2, "(MSE=", max(mse_cv[field]),")")
    field_optimal_K[field] = np.argmax(mse_cv[field])+2
   
mse_cv = pd.DataFrame(mse_cv)
mse_cv.index = mse_cv.index+2 # To get index = K
plt.plot(mse_cv)
plt.show()



# Let's try to impute the missing values
food_data_imputed = food_data_clean.copy().sample(10000)
food_data_imputed = food_data_imputed.dropna(axis=0, how='all', subset=numeric_columns)
before = food_data_imputed.isna().sum()
for row in food_data_imputed.iterrows():
    k_optimal = 10
    idx = row[0]
    values = row[1]
    nan_columns_numeric = [col for col in values.index[values.isna()].values if col in numeric_columns]
    nan_columns_not_numeric = [col for col in values.index[values.isna()].values if col not in numeric_columns]
    not_nan_columns_numeric = [col for col in values.index[~values.isna()].values if col in numeric_columns]
    columns_ = [col for col in not_nan_columns.values if col in numeric_columns]
    if not not_nan_columns_numeric:
        print("No value", row)
    else:
        scaler = StandardScaler().fit(food_ref_table[not_nan_columns_numeric])
        values_n = scaler.transform(values[not_nan_columns_numeric].values.reshape(1, -1).tolist())
        neighbors = NearestNeighbors(n_neighbors=K_max, metric='euclidean').fit(food_ref_table_n[not_nan_columns_numeric])

        food_data_imputed.loc[idx, not_nan_columns_numeric] = values_n[0]
        for col in nan_columns_numeric:
            k_optimal = field_optimal_K[col]
            indices = neighbors.kneighbors(values_n, n_neighbors=k_optimal, return_distance=False)[0]
            food_data_imputed.loc[idx, col] = food_ref_table_n[col].iloc[indices].mean()
        for col in nan_columns_not_numeric:
            food_data_imputed.loc[idx, col] = food_ref_table_n[col].iloc[indices].mode()[0]

after = food_data_imputed.isna().sum()
pd.concat([before, after], axis=1)

pca = PCA(n_components=5)
pca.fit(food_data_imputed[numeric_columns])
print(["{:.2%}".format(x) for x in pca.explained_variance_ratio_])

food_data_imputed_pca = pca.transform(food_data_imputed[numeric_columns])
food_data_imputed_pca = np.hstack((food_data_imputed_pca, food_data_imputed['nutrition_grade_fr'].values.reshape(-1, 1)))
food_data_imputed_pca = pd.DataFrame(food_data_imputed_pca, columns=['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'nutrition_grade_fr'])
food_data_imputed_pca[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']] = food_data_imputed_pca[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']].as_type('float64')
food_data_imputed_pca['nutrition_grade_fr'] = food_data_imputed_pca['nutrition_grade_fr'].as_type('category', ordered=True, categories=['a', 'b', 'c', 'd'])
food_data_imputed_pca.sort_values('nutrition_grade_fr', inplace=True)
sns.boxplot(x='nutrition_grade_fr', y='PCA_1', data=food_data_imputed_pca)
data = food_data_imputed[['energy_100g','proteins_100g', 'nutrition_grade_fr']]
sns.pairplot(data, hue='nutrition_grade_fr')


food_data_imputed['nutrition_grade_fr'].values.shape
    food_data_imputed_pca.shape
for each row:
    take the columns from the reference table where row has valid values
    normalize the row the scaler
    find the K=2 closest neighbours
    for all empty columns:
        if numeric: impute mean of neihgbors
        if categorical impute mode of neighbors:

np.array([1, 2]).reshape(1,-1).shape



food_data_imputed['nutrition_grade_fr'].values.reshape(-1, 1).shape
food_data_imputed_pca.shape

food_data_imputed['nutrition_grade_fr'].values.shape[0]



