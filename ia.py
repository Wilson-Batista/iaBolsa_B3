import dadosProcessar as dp
import joblib as job
import os
import warnings
#import numpy as np
from tensorflow.keras.models import Sequential as se, load_model as load
from tensorflow.keras.layers import Dense, LSTM, Dropout

#tratamento de erro de GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category = UserWarning)

class IA:

    #Méthodo construtor com caminho do escalonador salvo
    def __init__(self, nome_acao, dp_modulo = None):
        self.nome_acao = nome_acao
        self.modelo = None
        self.dp = dp_modulo # dadosProcessar module     taee11_scaler.pkl
        self.caminho_modelo= f"./modelos/{self.nome_acao}_modelo_lstm.keras"
        self.caminho_scalador = f"./modelos/{self.nome_acao}_scaler.pkl"
        
        # Busca carregar o modelo e o escalonador existentes na execução
        self.carregar_modelo_com_escalonador()
        print("modelo com escalonador executado ")
    
    def carregar_modelo_com_escalonador(self):
        #Carregando o modelo keras e o escalonador JobLib
        if os.path.exists(self.caminho_modelo) and os.path.exists(self.caminho_scalador):
            try:
                self.modelo = load(self.caminho_modelo)
                scala_load = job.load(self.caminho_scalador)
                #atualiza o escalonador casoo modulo dp tenha sido passado
                if self.dp:
                    self.dp.scala = scala_load
                    #self.dp.scala = job.load(self.caminho_modelo)
            except Exception as e:
                print(f"Erro ao carregar modelo ou escalonador para {self.nome_acao}: {e}")
                self.modelo = None
                if self.dp:
                    self.dp.scala = None
        else:
            print(f"O modelo ou o escalondaor  para {self.nome_acao} não foi encontrado, Treinando o meodelo")
            self.modelo = None
            if self.dp:
                self.dp.scala = None
                print()
    
    def construção_modelo_LSTM(self, input_shape):
        #Modelo LSTM
        modelo = se()
        neuronios = 100
        desligamento_neuronios = 0.2
        retorno_sequencial = True

        modelo.add(LSTM(units = neuronios, return_sequences=retorno_sequencial, input_shape=input_shape))

        #Dropout desligamento das unidades de neuronios no caso 20%, issso e elas começaram com 0
        modelo.add(Dropout(desligamento_neuronios))
        modelo.add(LSTM(units = neuronios,return_sequences=retorno_sequencial))
        modelo.add(Dropout(desligamento_neuronios))
        modelo.add(LSTM(units = neuronios))
        modelo.add(Dropout(desligamento_neuronios))
        #prever o proximo preço
        modelo.add(Dense(1))
        #construção do modelo Karem
        optimize = "adam"
        loss = "mean_squared_error"
        modelo.compile(optimizer = optimize,loss = loss)
        return modelo
    
    def treinar_modelo(self,X1, Y1, x2, y2, scala):
        #Treinar o modelo e salvar juntamente com o escalonador.
        
        if self.modelo is None:
            #Formata os dados para modelo 3D LSTM
            treino_Entrada = (X1.shape[1],1)
            self.modelo = self.construção_modelo_LSTM(input_shape=treino_Entrada)
        
        passagem = 50
        numero_amostra = 64
        nivel_Informacao = 1

        print(f"Iniciando treinamento do modelo para {self.nome_acao}...")
        #Treinando modelo
        X1_esperado = X1.reshape((X1.shape[0], X1.shape[1],1))
        x2_esperado = x2.reshape((x2.shape[0], x2.shape[1],1))

        self.modelo.fit(X1_esperado, Y1, epochs=passagem, batch_size=numero_amostra, validation_data = (x2_esperado, y2), verbose=nivel_Informacao)

        print("Treino efetuado com sucesso")
        #Salvar o modelo treinado
        try:
            #Garantia que o diretorio modelo exista
            os.makedirs("./modelos", exist_ok=True)
            self.modelo.save(self.caminho_modelo)
            job.dump(scala, self.caminho_scalador)
            print("Modelo salvo com sucesso")
        except Exception as e:
            print("Erro ao tentar salvar modelo ou escalonador para ação {self.nome_acao}: {e}")

        #atualização o scaler na classe dp apos treinamento
        if self.dp:
            self.dp.scala = scala
            print("Situação scala " , scala)
        
    def fazer_previsao(self, dados_entrada_normalizados):
        """
        Faz previsões usando o modelo carregado.
        dados_entrada_normalizados: Dados de entrada já normalizados.
        """
        if self.modelo is None:
            print("Modelo não carregado ou treinado. Não é possível fazer previsões.")
            return None
        
        # O modelo LSTM espera entrada 3D (amostras, timesteps, features)
        # Remodela a entrada se necessário (assumindo dados_entrada_normalizados é 2D)
        dados_para_previsao = dados_entrada_normalizados.reshape((dados_entrada_normalizados.shape[0], dados_entrada_normalizados.shape[1], 1))

        previsao_normalizada = self.modelo.predict(dados_para_previsao) # 
        return previsao_normalizada
