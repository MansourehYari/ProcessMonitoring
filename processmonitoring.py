#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install catboost


# In[ ]:


pip install flowoct


# In[ ]:


pip install odtlearn


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


# In[2]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    return summary


# In[3]:


file_loc = "df.xlsx"


# In[4]:


df = pd.read_excel(file_loc)


# In[5]:


df.head()


# In[6]:


df.info ()


# In[7]:


print ("Total number of rows in dataset = {}".format(df.shape[0]))
print ("Total number of columns in dataset = {}".format(df.shape[1]))


# In[8]:


result = resumetable(df)
result


# In[9]:


# Drop rows with missing values in 'cluster' column
data = df.dropna(subset=['cluster'])


# In[10]:


# Split the 'cluster' column
data = data.assign(cluster=data['cluster'].str.split(',')).explode('cluster')


# In[11]:


# Convert 'cluster' to numeric
data['cluster'] = pd.to_numeric(data['cluster'], errors='coerce')


# In[12]:


# One-hot encode the 'cluster' column
cluster_dummies = pd.get_dummies(data['cluster'], prefix='cluster')
data = pd.concat([data, cluster_dummies], axis=1)


# In[13]:


data.head ()


# In[14]:


data.shape


# In[15]:


# Define the filename where you want to save the Excel file
final = 'extracted_data.xlsx'

# Exporting the DataFrame to an Excel file
data.to_excel(final, index=False)


# In[16]:


file_loc = "extracted_data.xlsx"

df2 = pd.read_excel(file_loc)

df2.head ()


# In[17]:


target_cols = [col for col in df2.columns if col.startswith('cluster')]


# In[18]:


target_cols


# In[19]:


X = df2.drop(target_cols, axis=1)  # Features
y = df2[target_cols[1]]  # Target(s)


# In[20]:


from sklearn.preprocessing import LabelBinarizer
import numpy as np


# In[21]:


X.head()


# In[22]:


# Set 'ID' column as index if not already done
X.set_index('ID', inplace=True)


# In[24]:


# Assuming you've decided on a strategy to handle missing values; here's an example of filling missing values for categorical columns
X.fillna('unknown', inplace=True)  # For categorical data; for numerical data consider using .fillna(X.mean()) or similar

# Binarize all columns in X except 'ID' (which is now the index and not in columns)
columns_to_binarize = X.columns  # Since 'ID' is no longer a column, it's excluded already
for col in columns_to_binarize:
    lb = LabelBinarizer()
    # Ensure data type is string for categorical binarization; adjust as necessary for your specific data
    X[col] = lb.fit_transform(X[col].astype(str))

# Display the modified DataFrame
X.head()


# In[25]:


from sklearn.model_selection import train_test_split
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[26]:


print(X.dtypes)


# In[27]:


categorical_features_indices = np.where(X.dtypes != float)[0]
categorical_features_indices


# In[28]:


cat_features = [
    "Gender",
    "Cause",
    "Arrivalmode",
    "numberofhospitalization",
    "AIS"
]


# In[29]:


from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score


# In[30]:


model = CatBoostClassifier(
    custom_loss=[metrics.Accuracy()],
    random_seed=42,
    logging_level='Silent'
)


# In[31]:


model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
)


# In[32]:


cv_params = model.get_params()
cv_params.update({
    'loss_function': metrics.Logloss()
})
cv_data = cv(
    Pool(X, y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)


# In[33]:


print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']),
    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
    np.argmax(cv_data['test-Accuracy-mean'])
))


# In[34]:


predictions = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)
print(predictions[:10])
print(predictions_probs[:10])


# In[35]:


model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')
model_without_seed.fit(X, y, cat_features=categorical_features_indices)

print('Random seed assigned for this model: {}'.format(model_without_seed.random_seed_))


# In[36]:


print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))


# In[37]:


params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': metrics.Accuracy(),
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': False
}
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)


# In[38]:


model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool);

print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))
print('')

print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, best_model.predict(X_test))
))


# In[39]:


from IPython.display import Image

# Get SHAP values
shap_values = model.get_feature_importance(train_pool, fstr_type='ShapValues')

# Plot SHAP values
shap_values


# In[40]:


import matplotlib.pyplot as plt
import numpy as np

feature_importances = model.get_feature_importance()
#features = np.array(cat_features)  # Or however you have your features stored

# Sort features by importance
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[ ]:


pip install shap


# In[42]:


import shap

# Explain the model's predictions using SHAP
# This can be computationally intensive for large datasets
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Plot the SHAP values for the first instance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])


# In[43]:


shap.summary_plot(shap_values, X_train)


# In[44]:


shap.summary_plot(shap_values, X_train, plot_type="bar")


# **XGOOST Model**

# In[45]:


import xgboost as xgb

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X, y)

# Make predictions
xgb_predictions = xgb_model.predict(X)

# Evaluate XGBoost model performance
xgb_accuracy = accuracy_score(y, xgb_predictions)
print('XGBoost Model Accuracy: {:.4f}'.format(xgb_accuracy))


# In[46]:


# Initialize JS visualization code
shap.initjs()

# Create a Tree Explainer object that can calculate shap values
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for all of X
shap_values = explainer.shap_values(X)

# Plot summary plot using SHAP values
shap.summary_plot(shap_values, X, plot_type="bar")

# You can also create a more detailed dot plot
shap.summary_plot(shap_values, X)


# Automated Dataset Import and Model Training Function

# In[47]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_and_train_model(file_path, target_column, test_size=0.2, random_state=42):
    """
    Loads a dataset, trains an XGBoost model, and displays outputs and plot.

    :param file_path: Path to the dataset file.
    :param target_column: The name of the target column in the dataset.
    :param test_size: Fraction of the dataset to be used as test set.
    :param random_state: Random state for train-test split.
    """
    # Loading the dataset based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format!")

    # Dataset preparation
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model training
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Making predictions and computing accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Visualization: Feature Importance and Confusion Matrix
    xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


# In[ ]:


#load_and_train_model('path/to/your_dataset.csv', 'target_column_name')


# ***FLOWOCT* ALgorithm**

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from odtlearn.flow_oct import FlowOCT, BendersOCT
from odtlearn.utils.binarize import binarize


# In[60]:


X = df2.drop(target_cols, axis=1)  # Features
y = df2[target_cols[1]]  # Target(s)


# In[61]:


# Set 'ID' column as index if not already done
X.set_index('ID', inplace=True)


# In[62]:


# Assuming you've decided on a strategy to handle missing values; here's an example of filling missing values for categorical columns
X.fillna('unknown', inplace=True)  # For categorical data; for numerical data consider using .fillna(X.mean()) or similar

# Binarize all columns in X except 'ID' (which is now the index and not in columns)
columns_to_binarize = X.columns  # Since 'ID' is no longer a column, it's excluded already
for col in columns_to_binarize:
    lb = LabelBinarizer()
    # Ensure data type is string for categorical binarization; adjust as necessary for your specific data
    X[col] = lb.fit_transform(X[col].astype(str))

# Display the modified DataFrame
X.head()


# In[71]:


import pandas as pd

# Assume X is your DataFrame and 'ID' is set as the index
# Fill missing values
X.fillna('unknown', inplace=True)  # For categorical data
# For numerical data, consider using:
# X[numerical_columns] = X[numerical_columns].fillna(X.mean())

# Binarize categorical columns (convert all at once)
X = pd.get_dummies(X, drop_first=True)  # drop_first=True to avoid multicollinearity

# Display the modified DataFrame
X.head()


# In[72]:


# Initialize and fit the FlowOCT model
stcl = FlowOCT(depth=1, solver="CBC", time_limit=100)
stcl.fit(X, y)


# In[65]:


predictions = stcl.predict(X)

print(f"In-sample accuracy is {np.sum(predictions==y)/y.shape[0]}")


# In[66]:


stcl.print_tree()


# In[67]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Assuming X contains your features and y contains your target variable
# Replace 'X' and 'y' with your actual feature matrix and target array
# Fit the decision tree model
stcl = DecisionTreeClassifier()
stcl.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
tree.plot_tree(stcl, filled=True, feature_names=X.columns)  # Assuming X contains your feature matrix
plt.show()


# In[68]:


stcl_acc = BendersOCT(solver="CBC", depth=5, obj_mode="acc")
stcl_acc.fit(X, y)


# In[73]:


predictions = stcl_acc.predict(X)
print(f"In-sample accuracy is {np.sum(predictions==y)/y.shape[0]}")


# In[74]:


stcl_acc.print_tree()


# In[ ]:


import shap

# Assuming 'stcl' is your trained FlowOCT model and 'X' is the DataFrame used for training

# Initialize the SHAP explainer (assuming FlowOCT is similar to tree-based models)
try:
    explainer = shap.TreeExplainer(stcl)
    shap_values = explainer.shap_values(X)

    # Plot summary plot using SHAP values
    shap.summary_plot(shap_values, X, plot_type="bar")
    shap.summary_plot(shap_values, X)

except Exception as e:
    print("Error calculating SHAP values:", e)


# **Compare 3 models codes**

# In[ ]:


X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
#from flowoct import FloWOctClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve


# Train the models
catboost_model = CatBoostClassifier().fit(X_train, y_train)
xgboost_model = XGBClassifier().fit(X_train, y_train)
flowoct_model = FloWOctClassifier().fit(X_train, y_train)

# Make predictions on the test set
catboost_preds = catboost_model.predict(X_test)
xgboost_preds = xgboost_model.predict(X_test)
flowoct_preds = flowoct_model.predict(X_test)

# Numerical Comparison
print("Accuracy Scores:")
print(f"CatBoost: {accuracy_score(y_test, catboost_preds)}")
print(f"XGBoost: {accuracy_score(y_test, xgboost_preds)}")
print(f"FloWOct: {accuracy_score(y_test, flowoct_preds)}")

# Visual Comparison
# Confusion Matrix
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title("CatBoost Confusion Matrix")
plt.imshow(confusion_matrix(y_test, catboost_preds))
plt.subplot(1, 3, 2)
plt.title("XGBoost Confusion Matrix")
plt.imshow(confusion_matrix(y_test, xgboost_preds))
plt.subplot(1, 3, 3)
plt.title("FloWOct Confusion Matrix")
plt.imshow(confusion_matrix(y_test, flowoct_preds))
plt.show()

# ROC Curve
fpr_catboost, tpr_catboost, _ = roc_curve(y_test, catboost_model.predict_proba(X_test)[:, 1])
fpr_xgboost, tpr_xgboost, _ = roc_curve(y_test, xgboost_model.predict_proba(X_test)[:, 1])
fpr_flowoct, tpr_flowoct, _ = roc_curve(y_test, flowoct_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_catboost, tpr_catboost, label=f"CatBoost (AUC = {auc(fpr_catboost, tpr_catboost):.2f})")
plt.plot(fpr_xgboost, tpr_xgboost, label=f"XGBoost (AUC = {auc(fpr_xgboost, tpr_xgboost):.2f})")
plt.plot(fpr_flowoct, tpr_flowoct, label=f"FloWOct (AUC = {auc(fpr_flowoct, tpr_flowoct):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
precision_catboost, recall_catboost, _ = precision_recall_curve(y_test, catboost_model.predict_proba(X_test)[:, 1])
precision_xgboost, recall_xgboost


# In[ ]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from odtlearn.flow_oct import FlowOCT
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import numpy as np


# Load the dataset
file_path = 'df.xlsx'
df = pd.read_excel(file_path)

# Separate features and target
X = df.drop('cluster', axis=1)
y = df['cluster']

# Encode the target variable to consecutive integers
le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


file_loc = "extracted_data.xlsx"

df2 = pd.read_excel(file_loc)

df2.head ()

target_cols = [col for col in df2.columns if col.startswith('cluster')]


# In[ ]:


X = df2.drop(target_cols, axis=1) # Features
y = df2[target_cols[1]] # Target(s)


# In[ ]:


X.set_index('ID', inplace=True)


# In[ ]:


# Assuming you've decided on a strategy to handle missing values; here's an example of filling missing values for categorical columns
X.fillna('unknown', inplace=True)  # For categorical data; for numerical data consider using .fillna(X.mean()) or similar

# Binarize all columns in X except 'ID' (which is now the index and not in columns)
columns_to_binarize = X.columns  # Since 'ID' is no longer a column, it's excluded already
for col in columns_to_binarize:
    lb = LabelBinarizer()
    # Ensure data type is string for categorical binarization; adjust as necessary for your specific data
    X[col] = lb.fit_transform(X[col].astype(str))

# Display the modified DataFrame
print(X.head())


# In[ ]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Handle non-numeric values in the dataset
X_train = X_train.apply(lambda x: pd.to_numeric(x, errors='coerce'))
X_test = X_test.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Fill NaN values with a suitable value (e.g., mean or median)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())


# In[ ]:


# Train the models
catboost_model = CatBoostClassifier().fit(X_train, y_train)
#xgboost_model = XGBClassifier().fit(X_train, y_train)
flowoct_model = FlowOCT(depth=3, solver="CBC", time_limit=100).fit(X_train, y_train)

# Make predictions on the test set
catboost_preds = catboost_model.predict(X_test)
#xgboost_preds = xgboost_model.predict(X_test)
flowoct_preds = flowoct_model.predict(X_test)

# Numerical Comparison
print("Accuracy Scores:")
print(f"CatBoost: {accuracy_score(y_test, catboost_preds)}")
#print(f"XGBoost: {accuracy_score(y_test, xgboost_preds)}")
print(f"FloWOct: {accuracy_score(y_test, flowoct_preds)}")

print("\nF1 Scores:")
print(f"CatBoost: {f1_score(y_test, catboost_preds, average='weighted')}")
#print(f"XGBoost: {f1_score(y_test, xgboost_preds, average='weighted')}")
print(f"FloWOct: {f1_score(y_test, flowoct_preds, average='weighted')}")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Function to load and preprocess dataset
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    # Assuming ID is a column to set as index
    df.set_index('ID', inplace=True)
    lb = LabelBinarizer()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = lb.fit_transform(df[col].astype(str))
    return df

# Function to prepare data
def prepare_data(df, target_cols_prefix):
    target_cols = [col for col in df.columns if col.startswith(target_cols_prefix)]
    if not target_cols:
        raise ValueError(f"No columns start with {target_cols_prefix}")
    X = df.drop(columns=target_cols)
    y = df[target_cols[0]]  # Change here to use the first found target column
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models function
def train_and_evaluate(X_train, X_test, y_train, y_test):
    catboost_model = CatBoostClassifier(silent=True).fit(X_train, y_train)
    catboost_preds = catboost_model.predict(X_test)
    catboost_acc = accuracy_score(y_test, catboost_preds)
    catboost_f1 = f1_score(y_test, catboost_preds, average='weighted')

    xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    xgboost_preds = xgboost_model.predict(X_test)
    xgboost_acc = accuracy_score(y_test, xgboost_preds)
    xgboost_f1 = f1_score(y_test, xgboost_preds, average='weighted')

    print("Accuracy Scores:")
    print(f"CatBoost: {catboost_acc}")
    print(f"XGBoost: {xgboost_acc}")
    print("\nF1 Scores:")
    print(f"CatBoost: {catboost_f1}")
    print(f"XGBoost: {xgboost_f1}")

    return catboost_model, xgboost_model

# Function to plot evaluation curves
def plot_evaluation_curves(models, X_test, y_test):
    plt.figure(figsize=(15, 7))
    for model, label in models:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'{label}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
    plt.show()

# Main execution
file_path = "df.xlsx"
df = load_dataset(file_path)
X_train, X_test, y_train, y_test = prepare_data(df, 'cluster')
models = train_and_evaluate(X_train, X_test, y_train, y_test)
plot_evaluation_curves([(models[0], "CatBoost"), (models[1], "XGBoost")], X_test, y_test)


# In[ ]:


# Convert relevant columns to numeric, forcing errors to NaN
df['Mechanism_of_injury'] = pd.to_numeric(df['Mechanism_of_injury'], errors='coerce')
df['Cause_of_traumatic_injury'] = pd.to_numeric(df['Cause_of_traumatic_injury'], errors='coerce')
#  the correlation heatmap of numeric columns
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

