# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor #random forest
from sklearn.metrics import r2_score #r square sco

# df = pd.read_csv("linear_regression_dataset.csv", sep = ";")
# # df = pd.DataFrame(linear_regression_dataset.csv, columns = ["deneyim","maas"])
#
# plt.scatter(df.deneyim,df.maas)
# plt.xlabel("deneyim")
# plt.ylabel("maas")
# # plt.show()
#
#
# # sklearn
from sklearn.linear_model import LinearRegression
#
# linear_reg = LinearRegression()
#
# x = df.deneyim
# y = df.maas
# type(x)
#
# #numpy'a çevirdim
# x= df.deneyim.values
# y = df.maas.values
# print(x)
# print(x.shape) #sklearn böyle kullanım kabul etmior
#
# x = x= df.deneyim.values.reshape(-1,1)
# y = df.maas.values.reshape(-1,1)
# x.shape #(14,1)
#
# linear_reg.fit(x,y)
#
# b0 = linear_reg.predict([[0]])
# print("b0: ", b0)
#
# b0_ = linear_reg.intercept_  #b0 bulmak için, kesişim
# print("b0_: ", b0_," y eksenini kestiği nokta intercept")
#
# b1 = linear_reg.coef_
# print("b1: ", b1," eğim, slope")
#
# # maas = 1663 + 1138*deneyim
# maas_tahmin = b0_ + b1*11
# print("11 yıllık deneyim için maas tahmini: ", maas_tahmin)
# print("11 yıllık deneyim için maas tahmini kısa yol: ", linear_reg.predict([[11]]))
#
# array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  #deneyim
#
# plt.scatter(x,y,color="blue")
# y_head = linear_reg.predict(array)
# plt.xlabel("deneyim")
# plt.ylabel("maas")
#
# plt.plot(array, y_head, color ="red")
# plt.show()

###########################################################

# ### multiple linear regression
#
# #naas = dependent variable
# #deneyim =independent variable
#
# df = pd.read_csv("multiple_linear_regression_dataset.csv", sep = ";")
# x = df.iloc[:,[0,2]]
# y = df.maas.values.reshape(-1,1)
#
# multiple_linear_regression = LinearRegression()
# multiple_linear_regression.fit(x,y)
#
# #katsayılar bulundu
# print("b0: ", multiple_linear_regression.intercept_)
# print("b1, b2: ", multiple_linear_regression.coef_)
#
# print(multiple_linear_regression.predict(np.array([[10,35],[5,35]]))) #deneyim ve yaşa göre


###########################################################
#
# ### polynomial linear regression
#
# df = pd.read_csv("polynomial+regression.csv", sep = ";")
#
# y = df.araba_max_hiz.values.reshape(-1,1) #values arraye çeviriyor
# x = df.araba_fiyat.values.reshape(-1,1)
#
# plt.scatter(x,y)
# plt.ylabel("araba_max_hiz")
# plt.xlabel("araba_fiyat")
#
# #linear regression = y = b0 + b1*x
# #multiple linear regression = y = b0 + b1*x1 + b2*x2
#
# lr = LinearRegression()
# lr.fit(x,y) #en uygun line, bunu MSE'ye göre çekiyor
#
# #predict
# y_head = lr.predict(x)
#
# plt.plot(x,y_head, color = "red", label ="linear")
#
# from sklearn.preprocessing import PolynomialFeatures
# polynomial_regression = PolynomialFeatures(degree=2) # x^2 ye kadar al
#
# x_polynomial = polynomial_regression.fit_transform(x)
#
# linear_regression2 = LinearRegression()
# linear_regression2.fit(x_polynomial,y)
#
# y_head2 = linear_regression2.predict([[x_polynomial]])
# plt.plot(x,y_head2, color = "green", label ="poly")
# plt.legend()
# plt.show()
#

#######################################################################

# df = pd.read_csv("decision+tree+regression+dataset.csv", sep = ";")
#
# x = df.iloc[:, 0].values.reshape(-1, 1)
# y = df.iloc[:, 1].values.reshape(-1, 1)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(x,y)
#
# tree_reg.predict([[5.5]])
# x2 = np.arange(min(x), max(x), 0.01).reshape(-1,1)
# y_head = tree_reg.predict(x2)
#
# #visualize
# plt.ylabel("ucret")
# plt.xlabel("tribun level")
# plt.scatter(x,y, color = "red")
# plt.plot(x2 ,y_head, color = "green")
# plt.legend()
# plt.show()
#######################################################################
# from sklearn.ensemble import RandomForestRegressor
# df = pd.read_csv("random+forest+regression+dataset.csv", sep = ";", header = None)
#
# x = df.iloc[:, 0].values.reshape(-1, 1)
# y = df.iloc[:, 1].values.reshape(-1, 1)
#
# rf = RandomForestRegressor(n_estimators=100, random_state=42) #random_state aynı random alınmasını sağlıyor
# rf.fit(x,y)
# print("7.5 seviyesinde ne kadar olduğu: ", rf.predict([[7.5]]))
#
# x2 = np.arange(min(x), max(x), 0.01).reshape(-1,1)
# y_head = rf.predict(x2)
#
# # #visualize
# plt.ylabel("ucret")
# plt.xlabel("tribun level")
# plt.scatter(x,y, color = "red")
# plt.plot(x2 ,y_head, color = "green") #x2 yeni tahmin edillmek istenen değerler
#                                        #y_head tahmin sonuçları
# plt.legend()
# plt.show()
#######################################################################

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:06:37 2018

@author: user
"""
#
# # %% libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # %% read csv
# data = pd.read_csv("data.csv")
# data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
# print(data.info())
#
# y = data.diagnosis.values
# x_data = data.drop(["diagnosis"], axis=1)
#
# # %% normalization
# x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
#
# # (x - min(x))/(max(x)-min(x))
#
# # %% train test split
# from sklearn.model_selection import train_test_split
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# x_train = x_train.T
# x_test = x_test.T
# y_train = y_train.T
# y_test = y_test.T
#
# print("x_train: ", x_train.shape)
# print("x_test: ", x_test.shape)
# print("y_train: ", y_train.shape)
# print("y_test: ", y_test.shape)
#
#
# # %% parameter initialize and sigmoid function
# # dimension = 30
# def initialize_weights_and_bias(dimension):
#     w = np.full((dimension, 1), 0.01)
#     b = 0.0
#     return w, b
#
#
# # w,b = initialize_weights_and_bias(30)
#
# def sigmoid(z):
#     y_head = 1 / (1 + np.exp(-z))
#     return y_head
#
#
# # print(sigmoid(0))
#
# # %%
# def forward_backward_propagation(w, b, x_train, y_train):
#     # forward propagation
#     z = np.dot(w.T, x_train) + b
#     y_head = sigmoid(z)
#     loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
#     cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1]  is for scaling
#
#     # backward propagation
#     derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]  # x_train.shape[1]  is for scaling
#     derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]  # x_train.shape[1]  is for scaling
#     gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
#
#     return cost, gradients
#
#
# # %% Updating(learning) parameters
# def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
#     cost_list = []
#     cost_list2 = []
#     index = []
#
#     # updating(learning) parameters is number_of_iterarion times
#     for i in range(number_of_iterarion):
#         # make forward and backward propagation and find cost and gradients
#         cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
#         cost_list.append(cost)
#         # lets update
#         w = w - learning_rate * gradients["derivative_weight"]
#         b = b - learning_rate * gradients["derivative_bias"]
#         if i % 10 == 0:
#             cost_list2.append(cost)
#             index.append(i)
#             print("Cost after iteration %i: %f" % (i, cost))
#
#     # we update(learn) parameters weights and bias
#     parameters = {"weight": w, "bias": b}
#     plt.plot(index, cost_list2)
#     plt.xticks(index, rotation='vertical')
#     plt.xlabel("Number of Iterarion")
#     plt.ylabel("Cost")
#     plt.show()
#     return parameters, gradients, cost_list
#
#
# # %%  # prediction
# def predict(w, b, x_test):
#     # x_test is a input for forward propagation
#     z = sigmoid(np.dot(w.T, x_test) + b)
#     Y_prediction = np.zeros((1, x_test.shape[1]))
#     # if z is bigger than 0.5, our prediction is sign one (y_head=1),
#     # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
#     for i in range(z.shape[1]):
#         if z[0, i] <= 0.5:
#             Y_prediction[0, i] = 0
#         else:
#             Y_prediction[0, i] = 1
#
#     return Y_prediction
#
#
# # %% logistic_regression
# def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
#     # initialize
#     dimension = x_train.shape[0]  # that is 30
#     w, b = initialize_weights_and_bias(dimension)
#     # do not change learning rate
#     parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
#
#     y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
#
#     # Print test Errors
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
#
#
# logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=300)

# # %% sklearn with LR

# from sklearn.linear_model import LogisticRegression
#
# lr = LogisticRegression()
# lr.fit(x_train.T, y_train.T)
# print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))



#######################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression #sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor #decision tree
from sklearn.ensemble import RandomForestRegressor #random forest
from sklearn.metrics import r2_score #r square score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"], axis=1, inplace=True)
data.tail()
# M kötü huyulu tümör
# B iyi huylu tümör

#scatter plot
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean , M.texture_mean, color ="red",label="kotu", alpha = 0.3)
plt.scatter(B.radius_mean , B.texture_mean, color ="green",label="iyi" , alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values #numpy formatına çekti
x_data = data.drop(["diagnosis"], axis = 1)

#normalization
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#knn model
from sklearn.neighbors import kneighbors_graph

knn = kNeighborsClassifier(n_neighbors = 3) #n_neighbors = k
knn.fit(x_train,y_train)













