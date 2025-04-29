#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# In[2]:


# Set random seed for reproducibility
np.random.seed(42)


# In[15]:


# Define column names (ID, 9 features, class)
columns = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
           'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
           'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=columns, na_values='?')


# In[16]:


data.head()


# In[17]:


# Drop ID column
data = data.drop('ID', axis=1)


# In[18]:


# Handle missing values in Bare_Nuclei (replace with median)
data['Bare_Nuclei'] = data['Bare_Nuclei'].fillna(data['Bare_Nuclei'].median())

# Convert Class to binary: 2 (benign) → 0, 4 (malignant) → 1
data['Class'] = (data['Class'] == 4).astype(int)

# Select features and target
X = data.drop('Class', axis=1)
y = data['Class']


# In[19]:


# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[20]:


# Step 3: Create preprocessing and modeling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])


# In[21]:


# Step 4: Train the model
pipeline.fit(X_train, y_train)


# In[22]:


# Step 5: Make predictions and evaluate
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilities for positive class (malignant)


# In[23]:


# Evaluation metrics
accuracy = pipeline.score(X_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("ROC-AUC Score:", roc_auc)


# In[24]:


# Step 6: Tune threshold
thresholds = np.arange(0.1, 0.9, 0.1)
best_threshold = 0.5
best_f1 = 0
for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    report = classification_report(y_test, y_pred_threshold, output_dict=True)
    f1 = report['1']['f1-score']
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold}, Best F1-Score: {best_f1}")


# In[26]:


# Step 7: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Breast Cancer Classification')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_breast_cancer.png')
plt.show()
plt.close()


# In[27]:


# Step 8: Save processed dataset
data.to_csv('processed_breast_cancer.csv', index=False)


# In[ ]:




