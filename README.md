=======================================
PREDIÇÃO DE TENDÊNCIAS COM ARIMA
=======================================

Este repositório contém códigos para a predição de tendências utilizando modelos ARIMA e dados do Google Trends. O código pode ser utilizado para prever tendências de qualquer tema, com o tema padrão sendo "pedicure". Os códigos estão divididos em duas seções: **Modelo Local** e **Modelo Online**, permitindo testar o modelo tanto com dados locais (em um arquivo CSV) quanto ao utilizar a API do Google Trends diretamente.

---------------------------------------
ESTRUTURA DO REPOSITÓRIO
---------------------------------------

- **Modelo Local**: Contém o código que pode ser executado localmente, utilizando dados de um arquivo CSV de exemplo com o tema "pedicure". Essa pasta inclui um arquivo CSV que pode ser baixado e utilizado para testes, sem a necessidade de integrar com a API do Google Trends.
  
- **Modelo Online**: Contém o código que se conecta diretamente à API do Google Trends para baixar dados em tempo real. Não é necessário baixar manualmente os dados, pois o código acessa automaticamente a API para obter os dados mais recentes.

---------------------------------------
INSTRUÇÕES DE USO
---------------------------------------

### 1. MODELO LOCAL

**Objetivo**: Utilizar um arquivo CSV com dados do tema "pedicure" para testar o modelo ARIMA sem depender da API do Google Trends.

**Como Usar**:
1. Na pasta `Modelo Local`, há um arquivo CSV de exemplo com dados sobre "pedicure". Baixe este arquivo para utilizá-lo nos testes.
2. No código, altere o caminho do arquivo CSV para o local correto no seu computador onde o arquivo está armazenado. No código de exemplo, você verá a linha:
   ```python
   df = pd.read_csv('caminho/do/arquivo.csv')
   ```
   Substitua `'caminho/do/arquivo.csv'` pelo caminho real do seu arquivo CSV.
3. O modelo pode ser executado localmente para realizar predições de testes nos codigos com nome "testar arima". Além disso, você pode ajustar manualmente os parâmetros do modelo ARIMA (valores de `p`, `d`, `q`) nas variáveis globais.

### 2. MODELO ONLINE

**Objetivo**: Obter dados diretamente da API do Google Trends para realizar a previsão de tendências em tempo real.

**Como Usar**:
1. O código no modelo online se conecta diretamente à API do Google Trends, não sendo necessário baixar manualmente os dados.
2. Para obter os dados mais recentes sobre o tema de sua escolha (por exemplo, "pedicure"), o código irá acessar automaticamente a API do Google Trends.
3. No código, altere o nome do tema de interesse e o período de tempo, caso queira testar um tópico diferente de "pedicure". O código já está configurado para puxar os dados dos últimos 5 anos.
4. O caminho do arquivo CSV não precisa ser alterado, pois os dados são carregados diretamente pela API.

### 3. ALTERAÇÃO DE PARÂMETROS DO MODELO ARIMA

**Objetivo**: Modificar os parâmetros do modelo ARIMA para testar diferentes configurações de predição.

**Como Usar**:
- O modelo ARIMA possui três parâmetros principais: `p`, `d` e `q`.
- No código, esses valores são definidos nas variáveis globais.
- Você pode alterar manualmente esses valores para testar como eles afetam a precisão da predição.

---------------------------------------
CONTRIBUIÇÕES
---------------------------------------

Sinta-se à vontade para contribuir com melhorias no código! Caso encontre algum erro ou tenha sugestões, crie uma **issue** ou envie um **pull request**.
