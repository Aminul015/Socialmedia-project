 
print("Answer to the Question Number 2")
 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import model_selection
 
dataset = pd.read_csv('socialmedia.csv')
 
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values
 
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
 
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
 
y_pred = classifier.predict(x_test)
 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test, y_pred)
print("Accuracy")
print(acc)
from sklearn.metrics import classification_report
report= classification_report(y_test, y_pred)
print("Precision, Recall, F1-Score")
print(report)