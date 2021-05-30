# Projeto Restaurante com Machine Learning

## Introdução:
Este projeto visa a aplicar conhecimentos na área do Machine Learning através da criação e configuração de um modelo de previsão em Python e Tensorflow com o objetivo de prever o número de refeições servidas de um determinado restaurante 1 dia no futuro, utilizando para isso dados diários relativos à precipitação, temperatura e de refeições servidas no passado.

<br />

## Pré-requisitos:
- [Python](https://www.python.org/downloads/).
Bibliotecas necessárias:
  - Numpy;
  - Matplotlib;
  - Pandas;
  - Tensorflow;
  - Scikit-Learn;
  - Mysql.connector; 

- Editor de código fonte. 
Sugestão:
[Spyder](http://docs.spyder-ide.org/current/installation.html)

- Sistema de Gestão de Base de Dados. 
Sugestão:
[SQL Workbench](https://dev.mysql.com/downloads/workbench/)

<br />

## Ficheiros necessários:

### - Estrutura_SQL.sql

Ficheiro do tipo sql que representa a estrutura da base de dados utilizada neste projeto.

A base de dados **restaurantedb** é constituída por duas tabelas, dataset e previsao.

A tabela **dataset** é constituída por quatro colunas: *dia, refeições, tempo* e *chuva* que representam respetivamente o número de refeições servidas, a temperatura em graus Celsius e a precipitação em mm diária de Julho de 2017 a Março de 2020.


###### Primeiras 4 linhas da tabela **dataset**:
|      dia     |   refeições   |  tempo  |  chuva  |
|:------------:|:-------------:|:-------:|:-------:|
|  2017-07-29  |      149      |   19.3  |   0.00  |
|  2017-07-30  |      190      |   17.6  |   0.00  |
|  2017-08-01  |      167      |   17.1  |   0.10  |
|  2017-08-02  |      157      |   17.6  |  16.70  |

A tabela **previsao** é constituída por 2 colunas: *data* e *valor* que representam o valor da previsão futura de refeições servidas no restaurante - esta tabela é criada sem qualquer dados iniciais visto que estes serão adicionados à medida que o modelo é executado.

<br />

### - Configurar_Modelo.py

Ficheiro de código em Python que demonstra como é possível criar um RNN - Recurrent Neural Network - e como o treinar/testar de forma a ser possível calcular previsões futuras com ele. 

O tipo de modelo escolhido para este caso de estudo foi o GRU e os dados utilizados foram os da tabela **dataset**.

O propósito deste ficheiro é de permitir a configuração do modelo sempre que se necessitar - como é fornecido o ficheiro Modelo.keras não é obrigatória a execução deste script para o funcionamento deste projeto.

<br />

### - Modelo.keras

Ficheiro em formato keras que representa o modelo de machine learning já previamente treinado e testado.

É a partir deste modelo que são calculadas as previsões futuras.

<br />

### - Previsao.py

Ficheiro de código em python que demonstra como é possível utilizar um modelo já previamente treinado e testado para calcular previsões futuras.

Nas primeiras linhas de código, após a conexão à base de dados, os dados da tabela **dataset** são guardados numa dataframe da biblioteca Pandas. Após isso, é criada outra dataframe denominada **x_50** que contém apenas os últimos 50 valores da tabela **dataset**.

```python
df = pd.read_sql_query('''select dia, refeições, tempo, chuva from dataset''', db)
x_50 = df.iloc[-50:]
```

**Regra**: para calcular uma previsão no futuro utilizando um modelo de machine learning como é visto neste projeto, é necessário 'alimenta-lo' com vários valores. Assim, e aplicando a lógica deste caso de estudo, se foi inserido na tabela **dataset** o número de refeições, a temperatura e chuva do dia 30-11-2020, é necessário incluir as 49 linhas anteriores a essa data na tabela para calcular a previsão a partir da função predict(x) utilizada no código (não é preciso ser necessariamente os 50 últimos valores, pode ser um número à escolha). Por fim, o resultado guardado na variável predict, e fazendo jus ao exemplo anterior, representa o valor previsto de refeições que serão servidas pelo restaurante no dia 01-12-2020.

```python
newmodel = load_model('/app/Modelo.keras', custom_objects={'loss_mse_warmup': loss_mse_warmup})
predict = newmodel.predict(x)
```

<br />

## Regras:
- Instalar todos os pré-requisitos;   
- Fazer o download do projeto no formato .zip;
- Fazer a descompactação do arquivo .zip;
- Abrir e executar ficheiro "Estrutura_SQL" no Sistema de Gestão de Base de Dados;
- Executar script python "Previsao" para obter uma previsão 1 dia no futuro.

<br />

## Fontes:
Abaixo estão listados alguns tutorias utilizados para criar este código:

[Machine Learning e Tensorflow](https://colab.research.google.com/drive/1F_EWVKa8rbMXi3_fG0w7AtcscFq7Hi7B#forceEdit=true&sandboxMode=true)

[Modelos de previsão GRU e LSTM](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Criação de um RNN](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/23_Time-Series-Prediction.ipynb)

[Previsões futuras](https://github.com/rahulbhadani/medium.com/blob/master/01_12_2021/Saving_Model_TF2.ipynb)

<br />

## Autoria
- Irene Canelas : [@IreneCanelas](https://www.github.com/IreneCanelas)

Projeto realizado no âmbito de estágio profissional na Feralbyte coordenado por Hugo Freire.
