import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

sns.set()
# Reading the Data
df = pd.read_csv(r'C:\Users\ttroc\Downloads\creditcard.csv')
print('Shape of Dataframe is %s' %(df.shape,))
print(df.head())

# Plotting the different variables
corr = df.corr()
sns.heatmap(corr, linewidths=0.5, xticklabels=True, yticklabels=True)
plt.show()

# Data Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(df.info()) # All column dtype is float64
print('Total missing values per column: %s' %(df.isna().sum(),)) # No NaN values detected
X_full = df.drop(['Class'], axis=1)
X = pd.DataFrame(scaler.fit_transform(X_full.values), columns=X_full.columns)
y = df[['Class']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train.values.ravel())
predict = model.predict(X_test)

# Evaluation
precision, recall, _ = precision_recall_curve(y_test, predict)
auc_score = auc(recall, precision)
f1_score = f1_score(y_test, predict)
print(auc_score)
print(f1_score)

# Plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()