#DNN training
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fnum = h5py.File('ACC_1000_frames.h5', 'r')
intens = fnum['intens']
train_data = fnum['train_data']
labels = fnum['labels']
fft_maps = fnum['radial_fft_intens']
fnum.close()

x,y = np.indices((130,40))
x1 = x.ravel(); y1 = y.ravel()
nn = np.where(y1 < 10+0.25*x1)[0]
nn_0 = np.where(y1 > 10+0.25*x1)[0]
nnn = nn.shape[0]

fftmap0 = fft_maps[0][:,0::2][20:150,:40]
fftmap1 = fftmap0.ravel()[nn]


n_model = train_data.shape[0]
#train_data_nor = train_data*0
#for j in range(n_model):
#    train_data_nor[j] = train_data[j]/np.sum(train_data[j])


trainX = train_data[:800]
trainY = labels[:800]

testX = train_data[800:]
testY = labels[800:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(3430,)),
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    #tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')
])

#model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

model.fit(trainX, trainY, epochs=80, batch_size=30, validation_split = 0.2)
predict_Y = model.predict(testX)
Error = predict_Y - testY
EMS = np.sqrt(Error[:,0]**2 + Error[:,1]**2 + Error[:,2]**2)
np.sum(EMS)

model.save('ACC_model_X80')
#model = tf.keras.models.load_model('ACC_model_X80')
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]
