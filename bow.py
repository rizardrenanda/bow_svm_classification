import numpy as np
import pylab as pl
from sklearn import svm, datasets
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def Vehicle_label(s):
    it = {b'Car': 0, b'Motorcycle': 1}
    return it[s]

#it = {b'Fairing-Motorcycle': 0, b'Cruiser-Motorcycle': 1}
#it = {b'character-1': 0, b'character-2': 1, b'character-3': 2}

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
path = './harle_mobil.txt'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={3672: Vehicle_label})

x, y = np.split(data, indices_or_sections=(3672,), axis=1)  # x is data, y is label
x = x[:, 0:2]  # 0:4 means your feature
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.05, test_size=0.95)

# 3. Train SVM classifier
classifier = svm.SVC(C = 2, kernel = 'linear', decision_function_shape='ovr')  # ovr: on to many
clf = classifier.fit(train_data, train_label)
clf2 = classifier.fit(x, y)

# 4. Write prediction and recall
expected = train_label.ravel()
predicted = clf.predict(test_data)

print('Accuracy score : ', accuracy_score(train_label.ravel(), classifier.predict(train_data)))
print('Confusion matrix :\n', confusion_matrix(train_label.ravel(), classifier.predict(train_data)))
print('Classification report :\n', classification_report(train_label.ravel(), classifier.predict(train_data)))

print("Training sets：", classifier.score(train_data, train_label))
print("Testing sets：", classifier.score(test_data, test_label))


# 5. Bag of visual words
## 1.a setup BOW
bow_train   = cv2.BOWKMeansTrainer(8) # toy world, you want more.
bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )



# ONLY PLOTTING
fig, ax = plt.subplots()
# title for the plots
title = ('SVM Classifier ')
# Set-up grid for plotting.
X0, X1 = x[:, 0], x[:, 1]
xx, yy = make_meshgrid(X0, X1)

cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
grid_test = np.stack((xx.flat, yy.flat), axis=1)
grid_hat = classifier.predict(grid_test)
grid_hat = grid_hat.reshape(xx.shape)

plot_contours(ax, clf2, xx, yy, cmap=cm_light, alpha=0.8)
plt.pcolormesh(xx, yy, grid_hat, cmap=cm_light)
ax.scatter(x[:,0], x[:,1], c=y[:,0], s = 20, cmap=cm_dark, edgecolors='k')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
