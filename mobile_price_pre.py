#!/usr/bin/env python
# coding: utf-8

# # 1. Import `Libraries`

# In[1]:


# Data Analysis libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Machine Learning libs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, RandomizedSearchCV


# others
from pprint import pprint
import os

# uncomment below code if u need plt.show() in your code editor or any environment
# %matplotlib inline


# ### `CONSTANTS`

# In[2]:


TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
RANDOM_STATE = 42
N_SPLITS = 5 #for K-FOLD Cross Validation
# will keep (N_COMPONENTS * 100)% of information (eigenvalues)
# best value is between 80%-95%
N_COMPONENTS = 0.95


# # 2. Reading Data from CSV Files
# The dataset is divided into two parts: train and test.
# In this practice, we will focus solely on the `train.csv` file.
# 
# see dataset: [Kaggle -> Mobile-Price-Classification][1]
# 
# [1]: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification "dataset link in kaggle"

# In[3]:


df = pd.read_csv(TRAIN_FILE)
# df_test = pd.read_csv(TEST_FILE)


# # 3. Exploratory Data Analysis (EDA) with `Pandas`
# In this section, we will perform a brief exploratory data analysis (EDA) on the dataset using the Pandas library.

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


print(f'Columns (features): {len(df.columns)}')
print(f'Rows: {df.shape[0]}')


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


# check if dataset has null value
print(df.isnull().sum())


# In[10]:


# check if dataset has nan value (like string & etc.)
non_numeric_columns = df.select_dtypes(exclude='number').columns
print(f'non numeric columns count: {len(non_numeric_columns)}')


# # 4. Analyzing Data: Distribution, Correlation, and Feature Relationships with Price
# In this section, we will analyze the dataset to understand:
# 1. The distribution of numerical features.
# 2. The correlation between features.
# 3. The relationship between each feature and the target variable `mobile price`.
# 
# We will use visualizations and statistical methods to gain insights into the data.

# ## 1.4. check distribution of features

# In[11]:


df.hist(figsize=(16,16))


# In[12]:


# As you can see, the front camera megapixels (fc) have some outliers.
df['fc'].describe()


# In[13]:


# Also pixel height have some outliers.
df['px_height'].describe()


# In[14]:


# Screen Width of mobile in cm
df['sc_w'].describe()


# In[15]:


# Data is well distributed
df['price_range'].value_counts().plot(kind='pie', figsize=(14,9))


# ## 2.4. The correlation between features

# In[16]:


corr = df.corr()

fig, ax = plt.subplots(figsize = (14,9))
cax = ax.matshow(corr, cmap = 'RdBu')
fig.colorbar(cax)

ax.set_xticks(range(len(corr)), corr.columns, rotation = 'vertical')
ax.set_yticks(range(len(corr)), corr.columns)


# In[17]:


high_corr = corr.where((corr > 0.6) * (corr < 1))
high_corr


# In[18]:


# Convert the correlation matrix into a Series with MultiIndex 
# by stacking column labels into row indices. 
# This flattens the matrix and removes NaN values by default.
stacked_corr = high_corr.stack()
stacked_corr.index.set_names(['Row','Column'], inplace=True)
stacked_corr.name = 'Correlation'
print(stacked_corr)


# In[19]:


# convert Series to DataFrame
df_corr = stacked_corr.reset_index()
df_corr


# ## 3.4. The relationship between each feature and the target variable `mobile price`.

# In[20]:


df.plot(kind= 'scatter', x = 'ram', y = 'price_range')


# In[21]:


df.boxplot(column = 'price_range', by ='pc')


# In[22]:


sns.pairplot(df, vars= ['ram', 'battery_power','px_height','px_width'], hue='price_range')


# In[23]:


df.groupby('price_range').mean()


# In[24]:


plt.figure(figsize=(20,8))
sns.heatmap(df.groupby('price_range').mean(), annot=True, cmap='coolwarm')


# # 5. Data Preprocessing: Outlier Removal and Feature Selection
# In this section, we will perform the following preprocessing steps:
# 1. Remove outliers using the Interquartile Range (IQR) method.
# 2. Perform feature selection using Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA).
# 
# These steps will help us clean the data and reduce its dimensionality for better model performance.

# ## 1.5. Remove outliers using the Interquartile Range (IQR) method.

# ### `front camera`

# In[25]:


q1 = df['fc'].quantile(0.25)
q3 = df['fc'].quantile(0.75)

iqr = q3 - q1

# lower bonad of front camera can't be negative
lower_bound = max(0,q1 - (1.5 * iqr))
upper_bound = q3 + (1.5 * iqr)

print('Q1:',q1)
print('Q3:',q3)
print('IQR:',iqr)
print('lower bound:',lower_bound)
print('upper bound:',upper_bound)


df_filtered = df[(df['fc'] >= lower_bound) & (df['fc'] <= upper_bound)]


# In[26]:


df_filtered['fc'].describe()


# In[27]:


df['fc'].describe()


# In[28]:


print(df_filtered.shape[0] * 100 /df.shape[0],'percent of the data were preserved')


# ### `pixel height`

# In[29]:


q1 = df['px_height'].quantile(0.25)
q3 = df['px_height'].quantile(0.75)


iqr = q3 - q1
lower_bound = max(0, q1 - (1.5 * iqr))
upper_bound = q3 + (1.5 * iqr)

print('Q1:',q1)
print('Q3:',q3)
print('IQR:',iqr)
print('lower bound:',lower_bound)
print('upper bound:',upper_bound)

df_filtered = df_filtered[(df_filtered['px_height'] >= lower_bound) & (df_filtered['px_height'] <= upper_bound)]
print()
print(df_filtered.shape[0] * 100 /df.shape[0],'percent of the data were preserved')

df_filtered['px_height'].describe()


# ### `Screen Width` 

# In[30]:


q1 = df['sc_w'].quantile(0.25)
q3 = df['sc_w'].quantile(0.75)

iqr = q3 - q1
lower_bound = max(0, q1 - (1.5 * iqr))
upper_bound = q3 + (1.5 * iqr)

print('Q1:',q1)
print('Q3:',q3)
print('IQR:',iqr)
print('lower bound:',lower_bound)
print('upper bound:',upper_bound)

df_filtered = df_filtered[(df_filtered['sc_w'] >= lower_bound) & (df_filtered['sc_w'] <= upper_bound)]
print()
print(df_filtered.shape[0] * 100 /df.shape[0],'percent of the data were preserved')

df_filtered['sc_w'].describe()


# In[31]:


df_cleaned = df.copy()
df_x_cleaned = df_cleaned.drop(columns=['price_range'])
df_y_cleaned = df_cleaned['price_range']

df_x = df.drop(columns=['price_range'])
df_y = df['price_range']


# ### IQR for all `Features`

# In[32]:


# normalize data using MinMaxScaler
scaler = MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df_x_cleaned), columns=df_x_cleaned.columns)


# In[33]:


Q1 = df_scaled.quantile(0.25)
Q3 = df_scaled.quantile(0.75)

IQR = Q3 - Q1
lower_bound = Q1 - (IQR * 1.5)
upper_bound = Q3 + (IQR * 1.5)

# if negative, set 0
lower_bound = lower_bound.clip(lower=0)

df_x_cleaned = df_scaled[((df_scaled >= lower_bound) & (df_scaled <= upper_bound)).all(axis=1)]
df_y_cleaned = df_y_cleaned[((df_scaled >= lower_bound) & (df_scaled <= upper_bound)).all(axis=1)]
df_cleaned = pd.concat([df_x_cleaned, df_y_cleaned], axis= 1)
print()
print(df_x_cleaned.shape[0] * 100 /df.shape[0],'percent of the data were preserved')

df_x_cleaned.describe()


# In[34]:


print(df_x_cleaned.shape)
print(df_y_cleaned.shape)
print(df_cleaned.shape)


# ## 2.5. Perform feature selection using Principal Component Analysis (PCA) & Linear discriminant analysis (LDA).

# ### `before preprocessing`

# If you want to see the impact of preprocessing up to this point, take a look at the n_components value for the regular data. The data has been reduced to one dimension.

# In[35]:


pca = PCA(n_components= N_COMPONENTS)
pca.fit_transform(df_x)
print("Number of remaining features:", pca.n_components_)


# Even if we set n_components to 0999, the result doesn't change significantly.

# In[36]:


pca = PCA(n_components= 0.999)
pca.fit_transform(df_x)
print("Number of remaining features:", pca.n_components_)


# ### `after pre-processing`

# In[37]:


pca = PCA(n_components= N_COMPONENTS)
df_x_pca = pca.fit_transform(df_x_cleaned)
df_x_pca = pd.DataFrame(df_x_pca, columns=[f'PCA {i+1}' for i in range(int(pca.n_components_))])
print("Number of remaining features:", pca.n_components_)


# In[38]:


lda = LDA(n_components= 3)
df_x_lda = lda.fit_transform(df_x_cleaned, df_y_cleaned)
df_x_lda = pd.DataFrame(df_x_lda, columns=[f'LDA {i+1}' for i in range(3)])
print("Number of remaining features:", lda.classes_)


# # 6. Comparing Machine Learning Model Performance on Raw vs. Preprocessed Data
# 
# the performance of machine learning models is evaluated and compared on two datasets:
# 
#     1. Raw Data (Before Preprocessing): Data that has not undergone any transformation or preprocessing.
# 
#     2. Preprocessed Data: Data that has been processed using techniques such as normalization, outlier removal, dimensionality reduction (PCA), and etc.

# The goal is to assess the impact of data preprocessing on the accuracy, speed, and overall performance of machine learning models. Various machine learning algorithms (e.g., Linear Regression, Decision Trees, SVM, etc.) are applied, and the results are compared using metrics such as accuracy, execution time, and other relevant evaluation criteria.

# ### k-fold cross validation & evaluate function

# In[39]:


def evaluate_model_with_kfold(model, x, y, kf):

    fold_reports = []

    for train_index, test_index in kf.split(x):
        # Split the data into training and testing sets
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(x_train, y_train)

        # Predict on the test set
        y_pred = model.predict(x_test)

        # Generate classification report and store it
        report = classification_report(y_test, y_pred, output_dict=True)
        fold_reports.append(report)

    # Calculate average metrics across folds
    average_metrics = {
        "precision": np.mean([report["weighted avg"]["precision"] for report in fold_reports]).round(4),
        "recall": np.mean([report["weighted avg"]["recall"] for report in fold_reports]).round(4),
        "f1-score": np.mean([report["weighted avg"]["f1-score"] for report in fold_reports]).round(4),
        "accuracy": np.mean([report["accuracy"] for report in fold_reports]).round(4),
    }

    return average_metrics


# In[40]:


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state= RANDOM_STATE)


# ## 1.6. Decision Tree Classifier 

# In[41]:


def barplt(metrics: dict, title='Model Evaluation Metrics') -> None:
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x=metrics.keys(), y=metrics.values(), hue= metrics.keys(), palette="RdBu", width=0.4, legend=True)

    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.02, round(v,4), ha='center',fontsize=12, fontweight='bold')

    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0,1)
    plt.show()


# ### without `pre-processing`

# In[42]:


decision_tree = DecisionTreeClassifier(random_state= RANDOM_STATE)
df_metrics = evaluate_model_with_kfold(decision_tree, df_x, df_y, kf)

pprint(df_metrics)
barplt(df_metrics,title='DecisionTreeClassifier without pre-processiong')


# ### with `pre-processing`
# ### PCA

# In[43]:


decision_tree_pca = DecisionTreeClassifier(random_state=RANDOM_STATE)
pca_metrics = evaluate_model_with_kfold(decision_tree_pca,df_x_pca,df_y_cleaned, kf)
pprint(pca_metrics)

barplt(pca_metrics,'DecisionTreeClassifier (PCA)')


# In[44]:


decision_tree_lda = DecisionTreeClassifier(random_state=RANDOM_STATE)
lda_metrics = evaluate_model_with_kfold(decision_tree_lda,df_x_lda,df_y_cleaned, kf)
pprint(lda_metrics)
barplt(lda_metrics,'DecisionTreeClassifier (LDA)')


# ## 2.6. Random Forest Classifier

# In[45]:


rfc = RandomForestClassifier(random_state=RANDOM_STATE)
rfc_metrics = evaluate_model_with_kfold(rfc, df_x,df_y, kf)
pprint(rfc_metrics)
barplt(rfc_metrics,'RandomForestClassifier (without pre-processing)')


# In[46]:


rfc_pca = RandomForestClassifier(random_state= RANDOM_STATE)
rfc_metrics_pca = evaluate_model_with_kfold(rfc_pca, df_x_pca, df_y_cleaned, kf)
pprint(rfc_metrics_pca)
barplt(rfc_metrics_pca, 'RandomForestClassifier (PCA)')


# In[47]:


rfc_lda = RandomForestClassifier(random_state= RANDOM_STATE)
rfc_metrics_lda = evaluate_model_with_kfold(rfc_lda, df_x_lda, df_y_cleaned, kf)
pprint(rfc_metrics_lda)
barplt(rfc_metrics_lda, 'RandomForestClassifier (LDA)')


# ## 3.6. MLP

# In[48]:


param_dist = {
    'hidden_layer_sizes': [tuple(np.random.randint(15, 41, 7)) for _ in range(10)],  # 7 layers, each with a random number of neurons between 15 and 40
    'activation': ['relu', 'tanh', 'logistic'],  # Activation function for hidden layers
    'solver': ['adam', 'sgd', 'lbfgs'],  # Optimizer
    'alpha': np.logspace(-5, 0, 5),  # Regularization term (log scale)
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate
    'max_iter': [500, 1000]  # Maximum number of iterations
}


# ### `without pre-processing`

# In[49]:


mlp = MLPClassifier(random_state= RANDOM_STATE)

random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=20, cv= kf, scoring='accuracy', n_jobs=-1, random_state= RANDOM_STATE)

random_search.fit(df_x,df_y)


# In[50]:


print("Best Parameters:", random_search.best_params_)
print("Best Cross-validation Accuracy:", random_search.best_score_)

best_mlp = random_search.best_estimator_

best_mlp_metrics = evaluate_model_with_kfold(best_mlp, df_x, df_y, kf)
print(f"Best MLP Metrics: {best_mlp_metrics}")


# In[51]:


pprint(best_mlp_metrics)
barplt(best_mlp_metrics, 'MLP (without pre-processing)')


# ### with `PCA`

# In[52]:


mlp_pca = MLPClassifier(random_state= RANDOM_STATE)

random_search_pca = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=20, cv= kf, scoring='accuracy', n_jobs=-1, random_state= RANDOM_STATE)

random_search_pca.fit(df_x_pca, df_y_cleaned)


# In[53]:


print("Best Parameters:", random_search_pca.best_params_)
print("Best Cross-validation Accuracy:", random_search_pca.best_score_)

best_mlp_pca = random_search_pca.best_estimator_

best_mlp_metrics_pca = evaluate_model_with_kfold(best_mlp_pca, df_x_pca, df_y_cleaned, kf)
print(f"Best MLP Metrics: {best_mlp_metrics_pca}")


# In[54]:


pprint(best_mlp_metrics_pca)
barplt(best_mlp_metrics_pca, 'MLP (PCA)')


# ### with `LDA`

# In[55]:


mlp_lda = MLPClassifier(random_state= RANDOM_STATE)

random_search_lda = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=20, cv= kf, scoring='accuracy', n_jobs=-1, random_state= RANDOM_STATE)

random_search_lda.fit(df_x_lda, df_y_cleaned)


# In[56]:


print("Best Parameters:", random_search_lda.best_params_)
print("Best Cross-validation Accuracy:", random_search_lda.best_score_)

best_mlp_lda = random_search_lda.best_estimator_

best_mlp_metrics_lda = evaluate_model_with_kfold(best_mlp_lda, df_x_lda, df_y_cleaned, kf)
print(f"Best MLP Metrics: {best_mlp_metrics_lda}")


# In[57]:


pprint(best_mlp_metrics_lda)
barplt(best_mlp_metrics_lda, 'MLP (LDA)')

