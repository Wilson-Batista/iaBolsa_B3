import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

nome = "petr4"
dadosApi = f"/home/wilson/Área de Trabalho/iaBolsa_B3/dados api/{nome}.csv"
dataFrame = pd.read_csv(dadosApi)
#converção coluna data string em data
dataFrame["Data"] = pd.to_datetime(dataFrame["Data"]) 
#tornando o como indice da linha
dataFrame.set_index("Data",inplace=True)

#seleção da coluna preço de fechamento e muda para intervalo entre manor -1 e maior 1 
data = dataFrame["4. close"].values.reshape(-1, 1)

scala = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = scala.fit_transform(data)
#print(dados_normalizados)

# Criar sequências para treinamento
def criar_sequncia_treino(data, sequencia_fim):
    X, y = [], []
    for i in range(len(data)-sequencia_fim-1):
        X.append(data[i:(i+sequencia_fim), 0])
        y.append(data[i+sequencia_fim, 0])
    return np.array(X), np.array(y)

numero_dias = 90
dados_testes = 0.2
embaralhar = False
X, y = criar_sequncia_treino(dados_normalizados, numero_dias)
# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dados_testes, shuffle=embaralhar)

#X1 valores de entrada para terino, x2 valores utilizado para testar o modelo com dados novos.Y1 Valores que o modelo tentara prever, y2 valores alvos correspondentes a x2 que a ia prevera. y2 e usada para saber a performace
#print(dataFrame.head(50))
X1, x2, Y1, y2 = X_train,X_test,y_train,y_test
#print("X1 = " , X1, "X2 = " , x2, "Y1 = " , Y1, "Y2 = " , y2)

'''
print("As primeiras linhas do CSV")
print(dataFrame.head())
print("As ultimas linhas do CSV")
print(dataFrame.tail())
print("Informação dos dados")
print(dataFrame.info())
print("Estatisticas")
print(dataFrame.describe())
print("Valores Ausentes")
print(dataFrame.isnull())'
print(dataFrame.duplicated())


# Média movel Simples (SMA) de 3, 5 e 7 dias
dataFrame["média_movel_simples_3"] = dataFrame["4. close"].rolling(window=3).mean()
dataFrame["média_movel_simples_5"] = dataFrame["4. close"].rolling(window=5).mean()
dataFrame["média_movel_simples_10"] = dataFrame["4. close"].rolling(window=10).mean()
dataFrame["média_movel_simples_20"] = dataFrame["4. close"].rolling(window=20).mean()

#Média movel ponderada (SMA de 3, 5 e 7 dias)
dataFrame["média_movel_ponderada_3"] = dataFrame["4. close"].ewm(span=3, adjust=False).mean()
dataFrame["média_movel_ponderada_5"] = dataFrame["4. close"].ewm(span=5,adjust=False).mean()
dataFrame["média_movel_ponderada_10"] = dataFrame["4. close"].ewm(span=10,adjust=False).mean()
dataFrame["média_movel_ponderada_20"] = dataFrame["4. close"].ewm(span=20 , adjust=False).mean()

#remoção de dados NAN das medias
dataFrame = dataFrame.dropna(subset=["média_movel_simples_3","média_movel_simples_5", "média_movel_simples_10", "média_movel_simples_20", "média_movel_ponderada_3", "média_movel_ponderada_5", "média_movel_ponderada_10", "média_movel_ponderada_20"])
'''
