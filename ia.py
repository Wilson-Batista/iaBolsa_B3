import dadosProcessar as dp
from tensorflow.keras.models import Sequential as se
from tensorflow.keras.layers import Dense, LSTM, Dropout

#modelo 3D
trainoEntrada = dp.X1.reshape((dp.X1.shape[0], dp.X1.shape[1], 1))
trainoTeste = dp.x2.reshape((dp.x2.shape[0], dp.x2.shape[1],1))

#Modelo LSTM
modelo = se()
neuronios = 50
desligamento_neuronios = 0.2
retorno_sequencial = True
modelo.add(LSTM(units=neuronios, return_sequences=retorno_sequencial, input_shape=(trainoEntrada.shape[1],1)))
#Dropout desligamento das unidades de neuronios no caso 20%, issso e elas começaram com 0
modelo.add(Dropout(desligamento_neuronios))
modelo.add(LSTM(units=neuronios,return_sequences=retorno_sequencial))
modelo.add(Dropout(desligamento_neuronios))
modelo.add(LSTM(units=neuronios))
modelo.add(Dropout(desligamento_neuronios))
#prever o proximo preço
modelo.add(Dense(1))
#construção do modelo Karem
optimize="adam"
loss = "mean_squared_error"
modelo.compile(optimizer=optimize,loss=loss)

#treinamento
'''
PASSAGEM: quantidade de vezes que a ia ira estudadar
AMOSTRAS: quantidades de conjuntos de dados que ele ira receber em lote para estudar
NIVEL DE INFORAMÇÃO: tipo de informação apresentada durante estudo
'''
passagem = 50
numero_de_amaostras = 32
nivel_de_informacao = 1
historico = modelo.fit(dp.X1, dp.Y1, epochs=passagem, batch_size = numero_de_amaostras, validation_data=(dp.x2,dp.y2),verbose=nivel_de_informacao)