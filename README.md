


# Indicium Lighthouse

## Pré-requisitos

Antes de rodar o projeto, certifique-se de ter instalado:
- Python (versão recomendada: 3.8+)
  
## Instalando as Dependências

O projeto foi feito utilizando Python, Pandas, Seaborn, ScikitLearn e Matplotlib majoritariamente.

Foi escolhido seguir com o projeto com notebooks ipynb para facilitar a interação entre dataframes e variáveis...

Dentro do diretório do projeto, siga os passos:

1. Criar ambiente virtual (diferente a depender do sistema operacional) ```pythom -m venv .venv```
2. Ativar o ambiente virtual e instalar as dependências com ```pip install -r requirements.txt```
3. Usar o ambiente virtual como kernel para os notebooks ipynb.
4. Rodar as células do arquivo "challenge.ipynb" para fazer o preprocessamento e gerar os gráficos corretos para análise, além de realizar e validar testes de hipótese ANOVA, também gera o keywords.csv e desafio_indicium_imdb_after_process.csv, que é os dados após preprocessamento.
5. Rodas as células do arquivo "model.ipynb" para realizar a avaliação de modelos e ao final gerar a predição usando o GradientBoostingRegressor. Também salva os .pkl dos transformers para o MultilabelBinarizer e o próprio modelo.

## Arquivos

- model.pkl (GradientBoostingRegressor)
- multilabel_binaraizer_actors.pkl (Transformador para as colunas: Star1, Star2, Star3, Star4)
- multilabel_binarizer_genres.pkl (Transformador para a coluna de Genre)
- requirements.txt (Bibliotecas e versões)
- keywords.csv (Palavras chaves extraídas do Overview)
- challenge.ipynb (EDA, Testes de hipótese e gráficos)
- model.ipynb (Avaliação de diferentes algorítmos de regressão)
- LH_CD_PEDRO_HENRIQUE_BARBOSA_DA_CUNHA.pdf (Relatório final )
