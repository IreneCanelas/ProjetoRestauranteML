import mysql.connector as mysql
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.backend import square, mean
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import timedelta

db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "",
    database = "restaurantedb"
)

df = pd.read_sql_query('''select dia, refeições, tempo, chuva from dataset''', db)

df = df.iloc[1:]

x_50 = df.iloc[-50:]

data = x_50.iloc[-1]['dia']

data = data + timedelta(days=1)

x_50.set_index('dia', inplace=True)

y_50 = x_50['refeições']

y_50 = y_50.to_frame()

xscaler = MinMaxScaler()
xscaled = xscaler.fit_transform(x_50)
x = np.expand_dims(xscaled, axis=0) 

yscaler = MinMaxScaler()
yscaled = yscaler.fit_transform(y_50)
y = np.expand_dims(yscaled, axis=0) 

warmup_steps = 15

def loss_mse_warmup(y_true, y_pred):
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    mse = mean(square(y_true_slice - y_pred_slice))
    return mse

# Substituir '/app/Modelo.keras' pelo caminho onde se encontra guardado o Modelo
newmodel = load_model('/app/Modelo.keras', custom_objects={'loss_mse_warmup': loss_mse_warmup})
predict = newmodel.predict(x)

inverse = yscaler.inverse_transform(predict[0]) 

value = inverse[-1]
valor = value[0]
valor = round(valor)

cursor = db.cursor()

sql = """INSERT INTO previsao(data, valor) VALUES (%s, %s)"""
valores = (data, valor)

try:
    cursor.execute(sql, valores)
    db.commit()

except:
   db.rollback()

db.close()