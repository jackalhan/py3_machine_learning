import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('../breast-cancer-wisconsin.data')
# in file description-breast-cancer-wisconsin.names.txt
# There are 16 instances in Groups 1 to 6 that contain a single missing
# (i.e., unavailable) attribute value, now denoted by "?".
# and since handling number is more easier than string, we are going to replace ? with the -99999 to set the outlier basically.
df.replace('?', -99999, inplace=True) # modifies dataset right away.
# alternatively, we could drop these values with the df.dropna function.

# we need to remove attributes that are not related finding out the class. For example ID is one of them.
df.drop(['id'],1, inplace=True)

#features
X = np.array(df.drop(['class'],1)) #everything except class column
#labels
Y = np.array(df['class']) #just a class

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
classifier = svm.SVC()
classifier.fit(X_train, Y_train)

# we could save the classifier, pickle the classifier as a model.
accuracy = classifier.score(X_test, Y_test)

print(accuracy)

#predictions.
#we must be sure that, the following attributes are not in the original data.
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,10,10,2,3,2,1]])
prediction = classifier.predict(example_measures)
print(prediction)