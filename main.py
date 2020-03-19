from sklearn import svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def Faces_label(s):
    it = {b'Fairing-Motorcycle': 0, b'Cruiser-Motorcycle': 1}
    return it[s]

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# import some data to play with
path = './label.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={3672: Faces_label})

x, y = np.split(data, indices_or_sections=(3672,), axis=1)  # x is data, y is label
x = x[:, 0:2]  # 0:4 means your feature
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

# 3. Train SVM classifier
classifier = svm.SVC(kernel='linear')  # ovr: on to many
clf = classifier.fit(x, y)

fig, ax = plt.subplots()
# title for the plots
title = ('SVM ')
# Set-up grid for plotting.
X0, X1 = x[:, 0], x[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=20)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.xlim(xx.min(), xx.max())
ax.ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
