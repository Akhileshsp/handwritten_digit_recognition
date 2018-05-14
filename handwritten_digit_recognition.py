from sklearn.datasets import load_digits
from sklearn import svm
from matplotlib import pyplot as plt

digits = load_digits()

plt.gray()
plt.matshow(digits.images[50])
plt.show()

print(digits.images[50])

clf = svm.SVC()
clf.fit(digits.data[:-1], digits.target[:-1])
prediction = clf.predict(digits.data[50:51])
print("predicted digit -> ", prediction)
