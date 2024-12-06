# codigo para treinar o modelo com computação em paralelo.
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pytrends.request import TrendReq
import concurrent.futures
import random



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
    global valores_arima
    while True:
        try:
            p = random.randint(valores_arima[0][0], valores_arima[0][1])
            d = random.randint(valores_arima[2][0], valores_arima[2][1])
            q = random.randint(valores_arima[1][0], valores_arima[1][1])

            # p=10
            # d=1
            # q=30

            model = ARIMA(dados, order=(p, d, q))
            model_analize = model.fit()
            break
        except (ValueError, np.linalg.LinAlgError):
            print(f"Erro ao ajustar o modelo ARIMA: {p, d, q}. Reiniciando...")

    # Faz a previsão com o número de passos igual ao comprimento da avaliação
    forecast = model_analize.forecast(steps=len(avaliacao))

    return forecast, [p, d, q]


# Função para calcular a diferença entre previsão e dados reais.
def Resultados(avaliando, reais):
    global chave_trends
    distancia = sum(abs(reais[chave_trends][i] - avaliando[i]) for i in range(len(avaliando)))
    media = distancia / len(reais)

    return media

# Função de treinamento para rodar em paralelo.
def Treinamento(numero_treino):
    global treino, avaliacao
    previsoes_arima, parametros = Aplicando_Arima(treino)
    media_erro = Resultados(previsoes_arima, avaliacao)

    plt.plot(treino, label='Dados de Treino', color='blue')
    plt.plot(avaliacao, label='Dados de Avaliação', color='green')
    plt.plot(avaliacao.index, previsoes_arima, label='Previsão', color='red')
    plt.legend()
    plt.title(f'Previsão de Vendas - Execução {numero_treino}')
    plt.show()
    return (numero_treino, media_erro, parametros)

# Variáveis globais do programa.
chave_trends= "pedicure"
valores_arima = [[0, 30], [0, 30], [0, 2]]
melhores_individuos = []
num_instancias = 500
execucoes_por_rodada = 1
total_rodadas = num_instancias // execucoes_por_rodada
concluido = 0

# Criar dados de treino e avaliação.
dados = Pesquisa_Trends()
treino = dados[dados.index <= "2022-10-30 00:00:00"]
avaliacao = dados[(dados.index >= "2022-10-30 00:00:00") & (dados.index <= "2023-03-12 00:00:00")]




# Executa o treinamento em paralelo, 4 execuções por rodada.
for rodada in range(total_rodadas):
    with concurrent.futures.ProcessPoolExecutor() as executor:

        futuros = [executor.submit(Treinamento, i) for i in range(execucoes_por_rodada)]

        # Avalia os resultados e seleciona os melhores indivíduos.
        for futuro in concurrent.futures.as_completed(futuros):
            resultado = futuro.result()
            melhores_individuos.append(resultado)
            
            for futuro in concurrent.futures.as_completed(futuros):
                try:
                    resultado = futuro.result()  # Só acessa o resultado quando o processo termina
                    if resultado:
                        melhores_individuos.append(resultado)
                except Exception as e:
                    print(f"Erro ao obter o resultado: {e}")


            # Ordena e mantém apenas os 30 melhores indivíduos com menor erro.
            melhores_individuos = sorted(melhores_individuos, key=lambda x: x[1])[:30]

    # Atualiza o progresso
    concluido += execucoes_por_rodada
    progresso = (concluido / num_instancias) * 100
    print(f"Progresso: {progresso:.2f}% concluído")

# Exibe os 30 melhores indivíduos finais.
print("\n30 MELHORES INDIVÍDUOS:")
for melhor in melhores_individuos:
    print(melhor)