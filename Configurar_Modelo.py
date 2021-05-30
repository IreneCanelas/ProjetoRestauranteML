''' importar bibliotecas necessárias'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
import mysql.connector as mysql



''' ligação a base de dados mysql '''
db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "",
    database = "restaurantedb"
)

print(db)
df = pd.read_sql_query('''select dia, refeições, tempo, chuva from dataset''', db)



''' data de target para previsão '''
df.set_index('dia', inplace=True)

target = ['refeições']

shift_days = 1
shift_steps = shift_days * 1

df_targets = df[target].shift(-shift_steps)



''' transformar valores em arrays de NumPy '''
x_data = df.values[0:-shift_steps] 
y_data = df_targets.values[:-shift_steps] 
num_data = len(x_data) 

train_split = 0.9 
num_train = int(train_split * num_data)
num_test = num_data - num_train 

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1] 
num_y_signals = y_data.shape[1] 



''' escalar dados '''
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)



''' criar função que gera batches com valores de treino aleatórios '''
def batch_generator(batch_size, sequence_length):
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        yield (x_batch, y_batch)

batch_size = 32
sequence_length = 12 * 7 * 1

generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
x_batch, y_batch = next(generator)



''' set de validação '''
validation_data = (np.expand_dims(x_test_scaled, axis=0), np.expand_dims(y_test_scaled, axis=0))



''' criar Rede Neural '''
model = Sequential()
model.add(GRU(units=512, return_sequences=True, input_shape=(None, num_x_signals,)))

model.add(Dense(num_y_signals, activation='sigmoid'))
if False:
    from tensorflow.python.keras.initializers import RandomUniform
    init = RandomUniform(minval=-0.05, maxval=0.05)
    model.add(Dense(num_y_signals, activation='linear', kernel_initializer=init))



''' função de perda '''
warmup_steps = 15
def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    mse = mean(square(y_true_slice - y_pred_slice))
    return mse



''' compilar modelo '''
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()



''' funções de callback '''
path_checkpoint = 'Modelo.keras'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1,
                                      save_weights_only=False)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)

callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]



''' treinar modelo'''
model.fit(x=generator,
          epochs=5, 
          steps_per_epoch=50, 
          validation_data=validation_data,
          callbacks=callbacks)



''' load do checkpoint '''
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)



''' resultados do set de teste '''
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)



''' gerar previsões '''
def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test
    
    end_idx = start_idx + length
    
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    print(y_true)

    
    x = np.expand_dims(x, axis=0)

    y_pred = model.predict(x)

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    from sklearn.metrics import mean_absolute_error as mae
    print("\n Mean Absolute Error:", mae(y_true, y_pred_rescaled))
    
    from sklearn.metrics import mean_squared_error
    print("\n Mean Squared Error:", mean_squared_error(y_true, y_pred_rescaled))

    
    for signal in range(len(target)):
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        plt.figure(figsize=(15,5))
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        plt.ylabel(target[signal])
        plt.legend()
        plt.show()


plot_comparison(start_idx=0, length=1168, train=False)