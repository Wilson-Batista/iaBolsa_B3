import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#taee11; petr4,cple6
nome = "taee11"
#caminhos dados do ativo
dadosApi = f"/home/wilson/Área de Trabalho/iaBolsa_B3/dados api/{nome}.csv"
dataFrame = pd.read_csv(dadosApi)

#converção coluna data string em data
dataFrame["Data"] = pd.to_datetime(dataFrame["Data"]) 
#tornando o como indice da linha
dataFrame.set_index("Data",inplace=True)

#seleção da coluna preço de fechamento e muda para intervalo entre manor -1 e maior 1 
data = dataFrame["4. close"].values.reshape(-1, 1)

scala = MinMaxScaler(feature_range=(0, 1))
scala2 = scala
dados_normalizados = scala.fit_transform(data)
#print("dados normalizados classe dadosProcessar",dados_normalizados)
#print("scala classe dadosProcessar ", scala)

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

