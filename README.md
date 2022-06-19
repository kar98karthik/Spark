import pandas as pd
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.tree import export_graphviz

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print(data.feature_names)
print("\n")

df['target'] = data.target

#splitting traing and test dataset

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names],df['target'])
print(len(X_train))
print(len(X_test))

#DT = DecisionTreeClassifier()
DT = DecisionTreeClassifier(criterion="gini")

# Fit model
model = DT.fit(X_train, Y_train)

y_preds = model.predict(X_test)
print(y_preds)

print(data.feature_names)
print("\n")

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize =(8,5), dpi=500)
tree.plot_tree(DT,fontsize=7, feature_names=data.feature_names,class_names=data.target_names)

labels=[0,1,2]
cmx=confusion_matrix(Y_test,y_preds, labels)
print("\n Confusion Matrix : \n",cmx)

print("\n")
print(classification_report(Y_test, y_preds))
