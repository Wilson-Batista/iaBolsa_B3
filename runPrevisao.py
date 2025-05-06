import ia
import dadosProcessar as dados
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
#tratamento erro GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# Fazer previsões
previsao = ia.modelo.predict(dados.x2)
previsao = dados.scala.inverse_transform(previsao)
valor_atual = dados.scala.inverse_transform(dados.y2.reshape(-1, 1))


# Obter os últimos 'seq_length' dias dos dados de teste
dados_recentes = dados.x2[-1].flatten()  # Pega a última sequência usada no teste

print("Valores Real y2")
print(valor_atual)
print("#" * 30)

print("Valor previsto pela IA")
print(previsao)

#Avaliação IA quanto menor o resltado, melhor desempenho
erro_absoluto = mean_absolute_error(valor_atual, previsao)
erro_quadradtico_medio = mean_squared_error(valor_atual, previsao)
raiz_quadrada_de_erro_quadradtico_medio = np.sqrt(erro_quadradtico_medio)

valor = dados.nome

print(f"Ação Analisada: {valor}\n\n")

print(f"Desvio médio entre os valores previstos e os valores reais : {erro_absoluto:.2f} \n")

print(f"Média dos erros Elevado ao Quadrado : {erro_quadradtico_medio:.2f} \n")

print(f"raiz_quadrada_de_erro_quadradtico_medio: {raiz_quadrada_de_erro_quadradtico_medio:.2f}")

#dia = 5
#prever_proximos_dias(dia)
#print(previsao)