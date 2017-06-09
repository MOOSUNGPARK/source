# 데이터 로드

from sklearn import datasets

dataset = datasets.load_boston()
data, target = dataset.data, dataset.target

print(dataset)


'''
# len(data) --> 506    X 샘플수
# len(data[0]) --> 13  컬럼의 수 
# len(target) --> 506   Y 샘플수

- CRIM     per capita crime rate by town      
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.        
- INDUS    proportion of non-retail business acres per town        
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)    
- NOX      nitric oxides concentration (parts per 10 million)     
- RM       average number of rooms per dwelling       
- AGE      proportion of owner-occupied units built prior to 1940       
- DIS      weighted distances to five Boston employment centres     
- RAD      index of accessibility to radial highways       
- TAX      full-value property-tax rate per $10,000      
- PTRATIO  pupil-teacher ratio by town      
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town       
- LSTAT    % lower status of the population        
- MEDV     Median value of owner-occupied homes in $1000's 
'''


# 데이터 전처리(모든 값 0~1 사이로 스케일링)

from sklearn import preprocessing

data_scaler = preprocessing.MinMaxScaler()
target_scaler = preprocessing.MinMaxScaler()

data = data_scaler.fit_transform(data)
target = target_scaler.fit_transform(target)

# 훈련용, 테스트용 데이터셋 나누기

from sklearn.model_selection import train_test_split
from neupy import environment

environment.reproducible()

x_train, x_test, y_train, y_test = train_test_split(
    data, target, train_size=0.85
)

# 신경망 만들기

from neupy import algorithms, layers

cgnet = algorithms.ConjugateGradient(
    connection=[
        layers.Input(13),
        layers.Sigmoid(50),
        layers.Sigmoid(1),
    ],
    search_method='golden',
    show_epoch=25,
    verbose=True,
    addons=[algorithms.LinearSearch],
)

# 훈련시키기

cgnet.train(x_train, y_train, x_test, y_test, epochs=100)

# 그래프로 에러 확인

from neupy import plots
plots.error_plot(cgnet)

# 에러 체크

from neupy.estimators import rmsle

y_predict = cgnet.predict(x_test).round(1)
error = rmsle(target_scaler.inverse_transform(y_test),
              target_scaler.inverse_transform(y_predict))
print(error)

############################################################################

'''


# MNIST 로드 

from sklearn import datasets, model_selection
mnist = datasets.fetch_mldata('MNIST original')
data, target = mnist.data, mnist.target



# 데이터 전처리 

from sklearn.preprocessing import OneHotEncoder

data = data / 255.
data = data - data.mean(axis=0)

target_scaler = OneHotEncoder()
target = target_scaler.fit_transform(target.reshape((-1, 1)))
target = target.todense()


from neupy import environment
import numpy as np
from sklearn.model_selection import train_test_split

environment.reproducible()

x_train, x_test, y_train, y_test = train_test_split(
    data.astype(np.float32),
    target.astype(np.float32),
    train_size=(6. / 7)
)

import theano
theano.config.floatX = 'float32'

from neupy import algorithms, layers

network = algorithms.Momentum(
    [
        layers.Input(784),
        layers.Relu(500),
        layers.Relu(300),
        layers.Softmax(10),
    ],
    error='categorical_crossentropy',
    step=0.01,
    verbose=True,
    shuffle_data=True,
    momentum=0.99,
    nesterov=True,
)

network.architecture()

network.train(x_train, y_train, x_test, y_test, epochs=20)

from neupy import plots
plots.error_plot(network)

actual_values = np.array([1, 1, 1])
model1_prediction = np.array([0.9, 0.9, 0.4])
model2_prediction = np.array([0.6, 0.6, 0.6])

from neupy import estimators
estimators.binary_crossentropy(actual_values, model1_prediction)

estimators.binary_crossentropy(actual_values, model2_prediction)


from sklearn import metrics

y_predicted = network.predict(x_test).argmax(axis=1)
y_test = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

print(metrics.classification_report(y_test, y_predicted))

score = metrics.accuracy_score(y_test, y_predicted)
print("Validation accuracy: {:.2%}".format(score))






'''