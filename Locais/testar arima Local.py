# codigo para fazer testes unitarios do modelo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Função para ler o arquivo CSV
def carregar_dados_trends(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, delimiter=',', skiprows=1)
    df.columns = ['Semana', 'pedicure']
    df['Semana'] = pd.to_datetime(df['Semana'], errors='coerce')
    df = df.dropna(subset=['Semana'])
    df.set_index('Semana', inplace=True)
    return df

# Função para aplicar o ARIMA.
def Aplicando_Arima(dados):
    global p, d, q
    
    while True:
        try:

            model = ARIMA(dados, order=(p, d, q))
            model_analize = model.fit()
            break
        except (ValueError, np.linalg.LinAlgError):
            print(f"Erro ao ajustar o modelo ARIMA: {p, d, q}. Reiniciando...")

    # Faz a previsão com o número de passos igual ao comprimento da avaliação
    forecast = model_analize.forecast(steps=len(avaliacao))

    return forecast, [p, d, q]


# Função para calcular a diferença entre previsão e dados reais.
def Resultados(previsoes_arima, reais):
    global chave_trends
    distancia = sum(abs(reais[chave_trends][i] - previsoes_arima[i]) for i in range(len(previsoes_arima)))
    media = distancia / len(reais)

    return media

# Função de treinamento para rodar em paralelo.
def Treinamento():
    global treino, avaliacao
    previsoes_arima, parametros = Aplicando_Arima(treino)
    media_erro = Resultados(previsoes_arima, avaliacao)

    plt.plot(treino, label='Dados de Treino', color='blue')
    plt.plot(avaliacao, label='Dados de Avaliação', color='green')
    plt.plot(avaliacao.index, previsoes_arima, label='Previsão', color='red')
    plt.legend()
    plt.title(f'Previsão de Vendas - taxa de erro {media_erro}')
    plt.show()
    return (media_erro, parametros)

# Variáveis globais do programa.
chave_trends= "pedicure"
melhores_individuos = []
p= 18
d= 2
q=18


# Criar dados de treino e avaliação.
dados = carregar_dados_trends("C:\\Users\\User\\Documents\\codigos\\git\\pedicure trends.csv")
treino = dados[dados.index <= "2022-10-30 00:00:00"]
avaliacao = dados[(dados.index >= "2022-10-30 00:00:00") & (dados.index <= "2023-03-12 00:00:00")]


    
resultado = Treinamento()


print(resultado)