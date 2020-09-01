# Training a logistic regression classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np
# loading datasets

iris = datasets.load_iris()
X = iris['data'][:,3:]
y = (iris["target"] == 2).astype(np.int)

clf =  LogisticRegression()

clf.fit(X,y)


X_New = np.linspace(0,4,1000).reshape(-1,1)
print(X_New)

y_Prob = clf.predict_proba(X_New)
plt.title('Virginica')
plt.plot(X_New, y_Prob[:,1], "C1", label="virginica")
plt.show()