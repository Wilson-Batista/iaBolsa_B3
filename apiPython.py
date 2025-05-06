import requests as requisicao
import pandas as pd
import chave as key
#api: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETR4.SAO&interval=60min&apikey=key
#https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETR4.SAO&interval=60min&month=2015-01&outputsize=full&apikey=key

#'https://www.alphavantage.co/query?function=HT_PHASOR&symbol=IBM&interval=weekly&series_type=close&apikey=demo'

#

chave = key
simbolo = "petr4"
#5min, 15min, 30min, 60min
'''
intervalo = "60min"
mes = "01"
ano = "2015"
saida = "full"
'''
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={simbolo}&apikey={chave}'
resposta = requisicao.get(url)
data = resposta.json()
#print(data.keys())
horario = data['Time Series (Daily)']
df = pd.DataFrame(horario).T
df.index = pd.to_datetime(df.index)
df = df.astype(float)
df.to_csv(f'/home/wilson/Área de Trabalho/Trabalho/dados api/{simbolo}.csv', index_label='Data')
print("Dados salvos")
print(data)