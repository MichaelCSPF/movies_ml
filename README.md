## Sistema de Recomendação de Filmes

Este projeto foi desenvolvido com o objetivo de criar um sistema de recomendação de filmes utilizando técnicas de Machine Learning, semelhante aos sistemas usados por plataformas como Netflix e YouTube. O modelo foi construído em Python, utilizando a técnica de KNeighbors da biblioteca scikit-learn para identificar sugestões de filmes baseadas na similaridade.

# Intenções e Aplicações
O sistema de recomendação criado pode ser adaptado para diferentes contextos de negócios, como BI em farmácias ou outras áreas correlatas. Por exemplo, o modelo pode sugerir produtos aos clientes com base em suas compras anteriores e preferências, melhorar o gerenciamento de estoque e personalizar campanhas de marketing.

## Estrutura do Código
# Importação de Dados e Bibliotecas
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Ajuste de Colunas e Tratamento de Dados
filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
avaliacao = pd.read_csv('ratings.csv')

# Tratamento de Valores Nulos
filmes['QTD_AVALIACAO'] = pd.to_numeric(filmes['QTD_AVALIACAO'], errors='coerce')
mediana_lan = filmes['IDIOMA'].mode()[0]
mediana_vote = filmes['QTD_AVALIACAO'].median()
filmes['IDIOMA'] = filmes['IDIOMA'].fillna(mediana_lan)
filmes['QTD_AVALIACAO'] = filmes['QTD_AVALIACAO'].fillna(mediana_vote)

# Filtragem de Avaliações e Filmes
qt_avaliacoes = avaliacao['ID_USUARIO'].value_counts() > 800
y = qt_avaliacoes[qt_avaliacoes].index
avaliacao = avaliacao[avaliacao['ID_USUARIO'].isin(y)]

filmes_qt_avaliacoes = filmes['QTD_AVALIACAO'].value_counts() > 800
X = filmes_qt_avaliacoes[filmes_qt_avaliacoes].index
filmes = filmes[filmes['QTD_AVALIACAO'].isin(X)]

# Junção de DataFrames e Criação da Matriz Pivot
avaliacao_filmes = avaliacao.merge(filmes, on='ID_FILME')
filmes_pivot = avaliacao_filmes.pivot_table(columns='ID_USUARIO', index='TITULO', values='AVALIACAO')
filmes_pivot.fillna(0, inplace=True)

/* Combina os DataFrames e cria uma matriz pivot para representar a interação entre usuários e filmes. */

# Criação da Matriz Sparsa e Modelo de Recomendação
filmes_matrix = csr_matrix(filmes_pivot)
modelo = NearestNeighbors(algorithm='brute')
modelo.fit(filmes_matrix)
distancia, sugestao = modelo.kneighbors(filmes_pivot, n_neighbors=3)

/* Converte a matriz pivot em uma matriz sparsa e treina o modelo de KNeighbors para encontrar filmes semelhantes. */

# Criação do DataFrame de Sugestões e Exportação
titulos = filmes_pivot.index
df_sugestoes = pd.DataFrame(index=titulos, columns=['Filme', 'Indicado1', 'Indicado2', 'Indicado3'])

for i in range(len(titulos)):
    df_sugestoes.iloc[i, 0] = titulos[i]
    df_sugestoes.iloc[i, 1:] = titulos[sugestao[i]]

df_sugestoes.to_csv('sugestoes_de_filmes.csv', index=False)

Creditos:

Aula base: Nerd dos Dados
