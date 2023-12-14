import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

wine = pd.read_csv('data/winequality-red.csv')
print(wine.head(10))
print("wine.shape", wine.shape)
print("wine.isnull().sum()", wine.isnull().sum())

# data distribution plot
plt.figure(figsize=(8, 6))
plt.hist(wine['quality'], bins=range(1, 11), align='left', color='#880808', edgecolor='black')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality')
plt.xticks(range(1, 11))
plt.show()

# correlation matrix
correlation = wine.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, annot=True, cmap='Reds')
plt.show()

x = wine.drop('quality', axis=1)
# as recommend written in kaggle Tips
y = wine['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)

# model
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Accuracy', accuracy*100)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=y.unique(), yticklabels=y.unique(), annot_kws={'size': 20})
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# tree plot
plt.figure(figsize=(12, 8))
plot_tree(rfc)
plt.show()