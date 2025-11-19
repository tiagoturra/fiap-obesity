# Tech Challenge 4

## O Problema

Imagina que você foi contratado como cientista de dados de
um hospital e tem o desafio de desenvolver um modelo de Machine Learning
para auxiliar os médicos e médicas a prever se uma pessoa pode ter
obesidade.

A obesidade é uma condição médica caracterizada pelo acúmulo
excessivo de gordura corporal, a ponto de prejudicar a saúde. Esse problema
tem se tornado cada vez mais prevalente em todo o mundo, atingindo pessoas
de todas as idades e classes sociais. As causas da obesidade são multifatoriais
e envolvem uma combinação de fatores genéticos, ambientais e
comportamentais.

Utilizando a base de dados disponibilizada neste desafio em
`./res/obesity.csv`, desenvolva um modelo preditivo e crie um sistema preditivo para
auxiliar a tomada de decisão da equipe médica a diagnosticar a obesidade.

## Sobre o Dataset

Este conjunto de dados ajuda a estimar níveis de obesidade com base em hábitos alimentares,
histórico familiar e condição física. Ele inclui dados de indivíduos do México, Peru e Colômbia,
abrangendo 16 características relacionadas ao estilo de vida e à saúde, com um total de 2.111 registros.
Os rótulos classificam os níveis de obesidade, variando de baixo peso até diferentes tipos de obesidade.

A maior parte dos dados foi gerada por meio de técnicas sintéticas, enquanto uma parte foi coletada
diretamente de usuários por meio de uma plataforma web. O conjunto é útil para tarefas de classificação,
regressão e agrupamento (clustering).

Fontes:

* [UCI - Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
* [Kaggle](https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset)

## Dicionário de dados

A seguir está o dicionário de dados contendo a definição de cada coluna do arquivo `obesity.csv`.

| Coluna | Definição |
|-|-|
| Gender | Gênero |
| Age | Idade |
| Height | Altura em metros |
| Weight | Peso em quilogramas |
| family_history | Algum membro da família sofreu ou sofre de excesso de peso? <br>*(yes/no)* |
| FAVC | Você come alimentos altamente calóricos com frequência? <br>*(yes/no)* |
| FCVC | Frequência de consumo de vegetais. <br>*(escala de 1 a 3)* |
| NCP | Quantas refeições principais você faz diariamente? |
| CAEC | Frequência do consumo de alimento entre as refeições? <br>*(Never, Sometimes, Frequently, Always)* |
| SMOKE | Você fuma? <br>*(yes/no)* |
| CH2O | Quanta água você bebe diariamente? <br>*(escala de 1 a 3)* |
| SCC | Você monitora as calorias que ingere diariamente? <br>*(yes/no)* |
| FAF | Com que frequência você pratica atividade física? <br>*(escala de 0 a 3)* |
| TUE | Quanto tempo você usa dispositivos tecnológicos <br> como celular, videogame, televisão, computador e outros? <br>*(escala de 0 a 3)* |
| CALC | Com que frequência você bebe álcool? <br>*(Never, Sometimes, Frequently, Always)* |
| MTRANS | Qual meio de transporte você costuma usar? <br>*(Automobile, Bike, Motorbike, Public Transportation, Walking)*|
| **Obesity** <br> *(coluna alvo)* | Nível de obesidade <br>*(Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III)* |

## Pipeline de Machine Learning

O pipeline de machine learning, demonstrando toda as etapas de feature
engineering bem como o modelo escolhido com assertividade acima de 75% pode ser
encontrado no arquivo `./pipeline.ipynb`.

Este notebook possui comentários detalhados para cada etapa do processo, facilitando
sua leitura e interpretação.

## Deploy

O deploy do modelo foi realizado na plataforma *Streamlit*  e pode ser consultado através
do seguinte link:

[**Streamlit**](https://streamlit.io/)

## Visão Analítica

Utilizamos o *NOME_DO_SOFTWARE* para montar o painel analítico com os principais
insights obtidos com o estudo sobre obesidade.

O compartilhamento com a equipe pode ser realizado através do link:

[**DASHBOARD**](https://acme.com)

## Vídeo

O vídeo explicativo, mostrando toda a estratégia utilizada e apresentação do
sistema preditivo e do dashboard analítico em uma visão de negócio tem a duração em torno de 4min a 10min e foi disponibilizado juntamente com a entrega, na plataforma FIAP.
