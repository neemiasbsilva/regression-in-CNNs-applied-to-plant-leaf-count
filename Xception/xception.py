#!/usr/bin/env python
# coding: utf-8

# 
# # Regression in Convolution Neural Network Applied to Plant Leaf Counting
# 
# ## Using the Xception Architecture

# **Importing the Librarys for to Use at Convolution Neural Network**
# *   **numpy:** import the numpy library for make some scientif calcul
# *   **os & csv:** import the os and csv to open our dataset
# *   **skimage:** import the skimage library for make manipulation in image.
# *   **matplotlib.pyplot:** import the matplotlib to show the result of experiments.

# In[ ]:


import numpy as np
import os
import csv 
from skimage import io, transform
import matplotlib.pyplot as plt


# **Import the colab to mount**
# 
# **NOTE: If you not have GPU in you computer, I recomend upload your dataset to Google Drive and run in Colab**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# 
# **Creating OpenCsv Function, OpenImage Function, and map_classes to get the dataset**

# In[ ]:



def openCsv(wayFile):
    way_classes = []
    way_datas = []
    
    count = 0
    
    with open(wayFile, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            way_datas.append(row[0])
            way_classes.append(int(row[1]))
    return way_datas, way_classes
  
def openImage(way_path, way_image, width, height):
    X = []
    for i in range(0, len(way_image)):
        img = io.imread('%s%s' % (way_path, way_image[i]))
        img = img[...,:3]
        img = transform.resize(img,(width, height))
        X.append(img)
    return X
  
def map_classes(way_classes):
    m = {}
    y = np.zeros(len(way_classes))
    uc = np.unique(way_classes)
    
    for i in range(0, len(uc)):
        m[uc[i]] = i
    
    for i in range(0, len(way_classes)):
        y[i] = m[way_classes[i]]
        
    return y, m, uc


# **Run the function openCSV, openImage, and map_classes**
# 
# **Note: The each image have the 299 by 299 (width and height).

# In[ ]:


width = 299
height = 299


way_path = '/content/drive/My Drive/ColabNotebooks/A1_A2_A3_A4/'
way_image, way_classes = openCsv('/content/drive/My Drive/ColabNotebooks/A1_A2_A3_A4.csv')
X_datas = openImage(way_path, way_image, width, height)
#y_layers, m, uc = map_classes(way_classes)
y_layers = way_classes
X_datas = np.asarray(X_datas)
y_layers = np.asarray(y_layers)


print(y_layers.shape)
print(y_layers)


# **Import the keras library for adapt InceptionResNetV2 to Regression**
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop, SGD, Adam


# 
# **Split the Dataset in train/dev/test using Skelearn library.**
# 
# **NOTE: The coresponding number of image is train: 518, dev: 162, and test: 130)**

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_datas, y_layers, test_size=0.20, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

#x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
#x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1], x_val.shape[2]))

#y_train = np_utils.to_categorical(y_train, len(uc))
#Y_test = np_utils.to_categorical(y_test, len(uc))
#y_val = np_utils.to_categorical(y_val, len(uc))

print(X_datas.shape)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
#print(np.asarray(range(len(uc))))
#print(y_val[0,:])


# ## Adapting the Xception Architecture to Regression Problems

# In[ ]:


from keras.applications.xception import Xception
from keras.models import Model

model = Xception(weights='imagenet', include_top=True, input_shape=(299,299, 3))

x = model.get_layer(index=len(model.layers)-2).output

print(x)
x = Dense(1)(x)

model = Model(inputs=model.input, outputs=x)
model.summary()


# **Using RMSprop optimizer, mean absolute error with metrics, and mean square erro with loss**

# In[ ]:


opt = RMSprop(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])


# **Puting the model for fit**
# 
# **NOTE: The number of epochs is set to 100**

# In[ ]:


network_history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data= (x_val, y_val))


# ### Save the Model Trained

# In[ ]:


#model.save('/content/drive/My Drive/ColabNotebooks/AllmodeloRMSpropXception.h5')

model.save('/content/drive/My Drive/ColabNotebooks/Xception/modelXception.h5')


# ### Load the Model Trained

# In[ ]:


from keras.models import load_model

#model = load_model('/content/drive/My Drive/ColabNotebooks/AllmodeloRMSpropXception.h5')

model = load_model('/content/drive/My Drive/ColabNotebooks/Xception/modelXception.h5')


# ## Predict the Model Trained

# **Show the three better images and the three wrong image of the test set**

# In[ ]:



predictTest = model.predict(x_test, verbose=1)
predictTest = predictTest.reshape(predictTest.shape[0])
#predictTest = predictTest.astype('int32')
#print(x_test.shape)
#print(predictTest.shape)
#print(y_test.shape)
#print(np.round(predictTest))
#print(y_test)
#print(predictTest)

mae = np.abs(y_test - predictTest)
#print(mae)
pos = np.argsort(mae)

print(pos[-1])
print(pos[-2])
print(pos[-3])

#As três imagens piores preditas.
print("\tAs tres imagens que estao piores preditas")
img1 = plt.imshow(x_test[pos[-1]], interpolation='none')
plt.show()
print("Ground Truth: ",y_test[pos[-1]])
print("Prediceted: ",predictTest[pos[-1]])

img2 = plt.imshow(x_test[pos[-2]], interpolation='nearest')
plt.show()
print("Ground Truth: ",y_test[pos[-2]])
print("Prediceted: ",predictTest[pos[-2]])

img3 = plt.imshow(x_test[pos[-3]], interpolation='nearest')
plt.show()
print("Ground Truth: ",y_test[pos[-3]])
print("Prediceted: ",predictTest[pos[-3]])

#As três imagens que estao melhores preditas.
print("\tAs tres imagens que estao melhores preditas")
img1 = plt.imshow(x_test[pos[1]], interpolation='nearest')
plt.show()
print("Ground Truth: ",y_test[pos[1]])
print("Prediceted: ",predictTest[pos[1]])

img2 = plt.imshow(x_test[pos[2]], interpolation='nearest')
plt.show()
print("Ground Truth: ",y_test[pos[2]])
print("Prediceted: ",predictTest[pos[2]])

img3 = plt.imshow(x_test[pos[3]], interpolation='nearest')
plt.show()
print("Ground Truth: ",y_test[pos[3]])
print("Prediceted: ",predictTest[pos[3]])



# **Predict of train set**

# In[ ]:


predictTrain = model.predict(x_train, verbose=1)
predictTrain = predictTrain.reshape(predictTrain.shape[0])
#predictTrain = predictTrain.astype('int32')
print(x_train.shape)
print(predictTrain.shape)
print(y_train.shape)
print(np.round(predictTrain))
print(y_train)


# **Predict of train set**

# In[ ]:


predictVal = model.predict(x_val, verbose=1)
predictVal = predictVal.reshape(predictVal.shape[0])
#predictVal = predictVal.astype('int32')
print(x_val.shape)
print(predictVal.shape)
print(y_val.shape)
print(np.round(predictVal))
print(y_val)


# ### Using Metrics R^2, MAE, and MSE for evaluat the train/dev/test set

# **R², MAE and MSE for Train Set**

# In[ ]:


from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error

y_true = y_train
predict = predictTrain

r2 = r2_score(y_true, predict)
mae = median_absolute_error(y_true, predict)
mse = mean_squared_error(y_true, predict)
print("MSE \t MAE \t R2")
print(mse, "\t", mae,"\t", r2)


# **R², MAE and MSE for Dev Set**

# In[ ]:


from sklearn.metrics import r2_score, median_absolute_error, mean_squared_error

y_true = y_val
predict = predictVal

r2 = r2_score(y_true, predict)
mae = median_absolute_error(y_true, predict)
mse = mean_squared_error(y_true, predict)
print("MSE \t MAE \t R2")
print(mse, "\t", mae,"\t", r2)


# **R², MAE and MSE for Test Set**

# **R² score**

# In[ ]:


from sklearn.metrics import r2_score

y_true = y_test
predict = predictTest

r2_score(y_true, predict)


# **MAE score**

# In[ ]:


from sklearn.metrics import median_absolute_error

y_true = y_test
predict = predictTest

median_absolute_error(y_true, predict)


# **Mean Squared Error -- score**

# In[ ]:


from sklearn.metrics import mean_squared_error


y_true = y_test
predict = predictTest

mean_squared_error(y_true, predict)  


# ### Implementing the Scatter Graphics to plot the Coefficient of Determination

# In[ ]:


import matplotlib.pyplot as plt


N = y_test.shape
x = predictTest
y = y_test
colors = y_test
#area = np.pi * (10 * np.random.rand(162))**2  # 0 to 15 point radii
area = 80
#plt.title("\nXception\n", fontsize=18)
plt.xlabel("\nPredicted\n", fontsize=12)
plt.ylabel("\nGround Truth\n", fontsize=12)
marker_size=15
#plasma viridis hot
plt.scatter(x, y, s=area, c=colors, cmap='cool', alpha=0.5)
plt.gca().set_axis_bgcolor('white')

cbar= plt.colorbar()
cbar.set_label("Number of Leaves", labelpad=+1)

plt.show()


