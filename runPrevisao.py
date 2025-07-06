import ia
import dadosProcessar as dados
import os
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from tensorflow.keras.models import load_model as load

#tratamento erro GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

#instaciar casse IA para o uso dentro do try
instacia_IA = ia.IA(nome_acao = dados.nome, dp_modulo = dados)

#caminho do modelo salvo
#modelo_caminho = f"./modelos/{dp.nome}_modelo_lstm.keras"
#scalar_caminho = f"./modelos/{dp.nome}_scaler.pkl"
try:
    if instacia_IA.modelo is None or instacia_IA.dp.scala is None:
        raise Exception("Erro ao carregar modelo escalonador: Necesario fazer o treinameto")
    
    # Fazer previsões
    previsao_normalizada = instacia_IA.fazer_previsao(dados.x2)
    previsao = instacia_IA.dp.scala.inverse_transform(previsao_normalizada)
    valor_atual = instacia_IA.dp.scala.inverse_transform(dados.y2.reshape(-1,1))

    # Obter os últimos 'seq_length' dias dos dados de teste
    dados_recentes = dados.x2[-1].flatten()
    print("Valores Real y2")
    print(valor_atual)
    print("#" * 40)

    print("Valor previsto pela IA")
    print(previsao)

    #Avaliação IA quanto menor o resltado, melhor desempenho
    erro_absoluto = mean_absolute_error(valor_atual, previsao)
    erro_quadradtico_medio = mean_squared_error(valor_atual, previsao)
    raiz_quadrada_de_erro_quadradtico_medio = np.sqrt(erro_quadradtico_medio)
    nome_acao = dados.nome
    print(f"\nAção Analisada: {nome_acao}\n\n")
    print(f"Desvio médio entre os valores previstos e os valores reais : {erro_absoluto:.2f} \n")
    print(f"Média dos erros Elevado ao Quadrado : {erro_quadradtico_medio:.2f} \n")
    print(f"raiz_quadrada_de_erro_quadradtico_medio: {raiz_quadrada_de_erro_quadradtico_medio:.2f}")
    

except Exception as e:
    print(f"Erro no processo de carregamento ou previsão: {e}")
    print("Iniciando o treinamento do modelo...")
    print("dados.scalar valor",dados.scala)
    instacia_IA.treinar_modelo(dados.X1,dados.Y1,dados.x2,dados.y2, dados.scala2) #<====
    #instacia_IA.dp.scala = dados.scala
    print("Modelo treinado com sucesso! Tentando fazer previsões novamente...", dados.scala)

    if instacia_IA.dp:
        dados.scala = instacia_IA.dp.scala # none
        print("Modelo treinado")
    try:
        #print("passou aqui")
        previsao_normalizada = instacia_IA.fazer_previsao(dados.x2)
        #print("passou aqui ", dados.scala)
        previsao = instacia_IA.dp.scala.inverse_transform(previsao_normalizada)
        #print("passou aqui")
        valor_atual = instacia_IA.dp.scala.inverse_transform(dados.y2.reshape(-1,1))

        dados_recentes = dados.x2[-1].flatten()

        print("\nValores Reais y2 (após novo treinamento)")
        print(valor_atual)
        print("#" * 30)
        print("Valor previsto pela IA (após novo treinamento)")
        print(previsao)

        erro_absoluto = mean_absolute_error(valor_atual, previsao)
        erro_quadradtico_medio = mean_squared_error(valor_atual, previsao)
        raiz_quadrada_de_erro_quadradtico_medio = np.sqrt(erro_quadradtico_medio)

        nome_acao = dados.nome

        print(f"\nAção Analisada: {nome_acao}\n\n")
        print(f"Desvio médio entre os valores previstos e os valores reais : {erro_absoluto:.2f} \n")
        print(f"Média dos erros Elevado ao Quadrado : {erro_quadradtico_medio:.2f} \n")
        print(f"raiz_quadrada_de_erro_quadradtico_medio: {raiz_quadrada_de_erro_quadradtico_medio:.2f}")

    except Exception as e:
        print(f"Erro durante a previsão/avaliação mesmo após o treinamento: {e}")

    