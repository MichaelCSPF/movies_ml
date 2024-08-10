import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Importando os dados
filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
avaliacao = pd.read_csv('ratings.csv')

# Ajuste de colunas e tratamento de dados
avaliacao = avaliacao.drop(columns='timestamp', errors='ignore')
filmes = filmes[['id', 'original_title', 'original_language', 'vote_count']]
filmes = filmes.rename(columns={'id': 'ID_FILME', 'original_title': 'TITULO', 'original_language': 'IDIOMA', 'vote_count': 'QTD_AVALIACAO'})

# Tratamento de valores nulos
filmes['QTD_AVALIACAO'] = pd.to_numeric(filmes['QTD_AVALIACAO'], errors='coerce')
mediana_lan = filmes['IDIOMA'].mode()[0]
mediana_vote = filmes['QTD_AVALIACAO'].median()
filmes['IDIOMA'] = filmes['IDIOMA'].fillna(mediana_lan)
filmes['QTD_AVALIACAO'] = filmes['QTD_AVALIACAO'].fillna(mediana_vote)

# Filtragem de avaliações
qt_avaliacoes = avaliacao['ID_USUARIO'].value_counts() > 800
y = qt_avaliacoes[qt_avaliacoes].index
avaliacao = avaliacao[avaliacao['ID_USUARIO'].isin(y)]

# Filtragem de filmes
filmes_qt_avaliacoes = filmes['QTD_AVALIACAO'].value_counts() > 800
X = filmes_qt_avaliacoes[filmes_qt_avaliacoes].index
filmes = filmes[filmes['QTD_AVALIACAO'].isin(X)]
filmes.loc[:, 'ID_FILME'] = filmes['ID_FILME'].astype(int)

# Junção de DataFrames
avaliacao_filmes = avaliacao.merge(filmes, on='ID_FILME')

# Criação da matriz pivot
filmes_pivot = avaliacao_filmes.pivot_table(columns='ID_USUARIO', index='TITULO', values='AVALIACAO')
filmes_pivot.fillna(0, inplace=True)

# Criação da matriz sparsa
filmes_matrix = csr_matrix(filmes_pivot)

# Modelo de recomendação
modelo = NearestNeighbors(algorithm='brute')
modelo.fit(filmes_matrix)
distancia, sugestao = modelo.kneighbors(filmes_pivot, n_neighbors=3)

# Criação do DataFrame de sugestões
titulos = filmes_pivot.index
df_sugestoes = pd.DataFrame(index=titulos, columns=['Filme', 'Indicado1', 'Indicado2', 'Indicado3'])

for i in range(len(titulos)):
    df_sugestoes.iloc[i, 0] = titulos[i]
    df_sugestoes.iloc[i, 1:] = titulos[sugestao[i]]

# Exportar para CSV
df_sugestoes.to_csv('sugestoes_de_filmes.csv', index=False)
