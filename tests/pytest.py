import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

X_train ,X_test, y_train, y_test = train_test_split(data, digits.target,test_size=0.5,shuffle=False)
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def test_biased():
	pred1 = predictions[0]
	count=1
	for i in predictions[1:]:
		if pred1==i:
			count+=1
	assert count!=len(predictions)

def test_union():
	class_set = {pred for pred in predictions}
	assert len(class_set) == 10
