import numpy as np
import os
import problem
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
#from hypervolume import HyperVolume

# root_dir = "outputs_housing"
# file_generation = '{}/generation_number.npy'.format(root_dir)
# generation = np.load(file_generation)
# # fitness_score_list = []
# # active_nodes_list = []
# population = []
# for gen in range(0, generation+1):
#     file_pop = '{}/gen{}_pop.npy'.format(root_dir, gen)
#     population += np.load(file_pop).tolist()
# scores = []

# for individual in population:
#     scores.append(individual.fitness.values[0])
# sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]

# for i in range(1,sample_best.num_blocks+1):
#     curr_block = sample_best.skeleton[i]["block_object"]
#     print('curr_block isDead = ', curr_block.dead)
#     print(curr_block.active_nodes)
#     for active_node in curr_block.active_nodes:
#         fn = curr_block[active_node]
#         print(fn)
#         print('function at: {} is: {}'.format(active_node, fn))
# print("fitness:", sample_best.fitness.values)

# from keras import Sequential, optimizers, callbacks
# from keras.layers import Dense

# model = Sequential()
# model.add(Dense(input_shape = problem.x_train[0].shape[1:], units = 128, activation = "relu"))
# for i in range(6):
#     model.add(Dense(units = 128, activation = "relu"))
# model.add(Dense(units = 1))
# adam = optimizers.Adam()
# model.compile(loss='mean_squared_error', optimizer=adam)
# earlStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
# history = model.fit(problem.x_train[0], problem.y_train, validation_data = (problem.x_val, problem.y_val), batch_size = 256, epochs = 100, callbacks = [earlStop])

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')

# predictions = model.predict(problem.x_test[0])
# predictions = problem.scalerY.inverse_transform(predictions)

# y_true = problem.scalerY.inverse_transform(problem.y_test)


# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from math import sqrt

Ourmse = 138540.84333948712# sqrt(mean_squared_error(y_true, predictions))
print("Mean Absolute error on un-normed data:", Ourmse)




"""Another Person's code: https://www.kaggle.com/yusukekawata/house-price-prediction-with-neural-net"""


#モジュールの読み込み
# from __future__ import print_function

# import pandas as pd
# from pandas import Series,DataFrame

# #from sklearn import svm
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# #from sklearn.metrics import accuracy_score

# import numpy as np
# import matplotlib.pyplot as plt

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam

# from sklearn.metrics import mean_squared_error

# path = "housing_dataset/kc_house_data.csv"
# #CSVファイルの読み込み
# data_set = pd.read_csv(path)

# #説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
# x_train,x_test,y_train,y_test = problem.x_train[0], problem.x_test[0], problem.y_train, problem.y_test

# #データの整形
# x_train = x_train.astype(np.float)
# x_test = x_test.astype(np.float)

# #ニューラルネットワークの実装①
# model = Sequential()

# model.add(Dense(50, activation='relu',input_shape = problem.x_train[0].shape[1:]))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Dense(50, activation='relu', input_shape=(50,)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Dense(50, activation='relu', input_shape=(50,)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Dense(1))

# model.summary()
# print("\n")

# #ニューラルネットワークの実装②
# model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse'])

# #ニューラルネットワークの学習
# history = model.fit(x_train, y_train,batch_size=200, epochs=100, verbose=1, validation_data=(x_test, y_test))

# #RMSE用にMSEを算出
# predictions2 = model.predict(x_test)
# predictions2 = problem.scalerY.inverse_transform(predictions2)


# mse = sqrt(mean_squared_error(y_true, predictions2))
competition1MSE = 136461.48 #mse
print("KERAS REG MSE : %.2f" % competition1MSE)




# y_pred = sca

"""https://www.kaggle.com/harlfoxem/house-price-prediction-part-2"""
LinearRegression =  169561.235885
Knn = 184936.008654
lasoRegression = 171369.075713
ridgeRegression = 171566.791859


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('LinReg', 'Knn', 'LRegr', 'rRegression', 'ezCGP', "regNeurl")
y_pos = np.arange(len(objects))
performance = [LinearRegression,Knn,lasoRegression,ridgeRegression,Ourmse, competition1MSE]
 
plt.bar(y_pos, performance, width = 1, color = ["b", "g", "r", "c", "m", "b"])
plt.xticks(y_pos, objects)
plt.ylabel('RMSE')
plt.title('Benchmark on Housing Dataset')
 
plt.show()
