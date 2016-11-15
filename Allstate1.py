
# import my libraries
import numpy
import warnings
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from xgboost import XGBRegressor

# load train and test data
# inspect features
train = pandas.read_csv("input/train.csv")
test = pandas.read_csv("input/test.csv")
print(train.shape)
print(test.shape)
# record the data id and drop it
test_ID = test['id']
train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)

# preprocessing
cat_size = 116
cont_size = 14
train["loss"] = numpy.log1p(train["loss"])

# one hot encoding of categorical data
labels = []
for i in range(cat_size):
    labels.append(list(set(train.iloc[:, i].unique()) | set(test.iloc[:, i].unique())))
train_cats = []
test_cats = []
for i in range(cat_size):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    train_feature = label_encoder.transform(train.iloc[:, i])
    train_feature = train_feature.reshape(train.shape[0], 1)
    test_feature = label_encoder.transform(test.iloc[:, i])
    test_feature = test_feature.reshape(test.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse = False, n_values = len(labels[i]))
    train_feature = onehot_encoder.fit_transform(train_feature)
    train_cats.append(train_feature)
    test_feature = onehot_encoder.fit_transform(test_feature)
    test_cats.append(test_feature)
    del train_feature
    del test_feature
    del onehot_encoder
train_cats = numpy.column_stack(train_cats)
test_cats = numpy.column_stack(test_cats)
print(train_cats.shape)
train_encoded = numpy.concatenate((train_cats, train.iloc[:, cat_size:].values), axis = 1)
print(train_encoded.shape)
print(test_cats.shape)
test_encoded = numpy.concatenate((test_cats, test.iloc[:, cat_size:].values), axis = 1)
print(test_encoded.shape)
del train
del test
del train_cats
del test_cats

# separate train data into train and validation parts

row, col = train_encoded.shape
features = train_encoded[:, 0 : col - 1]
loss = train_encoded[:, col - 1]
valid_size = 0.1
rseed = 0
x_train, x_valid, y_train, y_valid = cross_validation.train_test_split(features, loss, test_size = valid_size, random_state = rseed)
X = numpy.concatenate((x_train, x_valid), axis = 0)
Y = numpy.concatenate((y_train, y_valid), axis = 0)
n_estimators = 1000
best_model = XGBRegressor(n_estimators = n_estimators, seed = rseed)
best_model.fit(X, Y)

# make prediction and write to file
test_prediction = numpy.expm1(best_model.predict(test_encoded))

with open("submission.csv", 'w') as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(test_prediction)):
        subfile.write("%s,%s\n"%(test_ID[i], pred))

# delete all tables
del train_encoded
del test_encoded
del best_model
