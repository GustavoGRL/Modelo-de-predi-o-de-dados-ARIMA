# codigo para fazer testes unitarios do modelo

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pytrends.request import TrendReq



# Função para pegar dados do Google Trends.
def Pesquisa_Trends():
    global chave_trends
    pytrends = TrendReq(hl='pt-Br', tz=360)
    pytrends.build_payload([chave_trends], timeframe='today 5-y')
    data = pytrends.interest_over_time()
    if not data.empty:
        data = data.drop(columns=['isPartial'])
        return data
    else:
        print(f"Não foram encontrados dados para a palavra-chave '{chave_trends}'.")
        return False

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
p=30
d= 2
q=18
melhores_individuos = []
num_instancias = 2


# Criar dados de treino e avaliação.
dados = Pesquisa_Trends()
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