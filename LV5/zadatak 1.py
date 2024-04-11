import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a)

plt.scatter(X_train[:,0],X_train[:,1], c="green")
plt.scatter(X_test[:,0],X_test[:,1], c="red", marker='x')
plt.show()

# b)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c)

a0 = LogRegression_model.intercept_[0]
a1,a2 = LogRegression_model.coef_.T

b = -a2/a1
c = -a0/a1

plt.scatter(X_train[:,0],X_train[:,1], c="green")
plt.plot(X_train[:,0],b*X_train[:,0]+c)
plt.show()

# d)

y_test_p = LogRegression_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()
print(classification_report(y_test , y_test_p))

# e)

true = np.where(y_test_p == y_test)[0]
false = np.where(y_test_p != y_test)[0]

plt.figure()
plt.scatter(X_test[true,0], X_test[true,1], c = "green")
plt.scatter(X_test[false,0], X_test[false,1], c = "black")
plt.show()
