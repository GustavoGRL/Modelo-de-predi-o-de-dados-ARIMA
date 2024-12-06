#codigo usado para fazer treinamento individual.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random



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
    global valores_arima
    
    while True:
        try:
            p = random.randint(valores_arima[0][0], valores_arima[0][1])
            d = random.randint(valores_arima[2][0], valores_arima[2][1])
            q = random.randint(valores_arima[1][0], valores_arima[1][1])

            # p=18
            # d=2
            # q=18

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
    plt.title(f'Previsão de Pedicure - taxa de erro {media_erro}')
    # Ajuste da frequência e rotação dos ticks do eixo x
    plt.xticks(rotation=45)  # Rotaciona as datas para 45 graus
    
    # Define a frequência dos ticks (ajuste conforme o intervalo do seu dataset)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Mostra no máximo 10 rótulos no eixo x
    
    # Formata as datas (ajuste conforme necessário)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()  # Ajusta layout para evitar sobreposição
    plt.show()
    
    
    return (media_erro, parametros)

# Variáveis globais do programa.
chave_trends= "pedicure"
valores_arima = [[0, 30], [0, 30], [0, 2]]
melhores_individuos = []
num_instancias = 500


# Criar dados de treino e avaliação.
dados = carregar_dados_trends("C:\\Users\\User\\Documents\\codigos\\git\\pedicure trends.csv")
treino = dados[dados.index <= "2022-10-30 00:00:00"]
avaliacao = dados[(dados.index >= "2022-10-30 00:00:00") & (dados.index <= "2023-03-12 00:00:00")]



for rodada in range(num_instancias):
    
    resultado = Treinamento()
    melhores_individuos.append(resultado)
    melhores_individuos = sorted(melhores_individuos, key=lambda x: x[0])[:30]
    
    
    # Atualiza o progresso
    progresso = (rodada / num_instancias) * 100
    print(f"Progresso: {progresso:.2f}% concluído")

# Exibe os 30 melhores indivíduos finais.
print("\n30 MELHORES INDIVÍDUOS:")
for melhor in melhores_individuos:
    print(melhor)