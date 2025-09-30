# 🚀 Documentação - Predição de Sucesso de Startups

## 📋 Índice
- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Arquivos de Dados](#arquivos-de-dados)
- [Metodologia](#metodologia)
- [Funcionalidades do Código](#funcionalidades-do-código)
- [Execução](#execução)
- [Resultados](#resultados)

---

## 🎯 Visão Geral

Este projeto foi desenvolvido para uma competição Kaggle que visa prever o sucesso de startups. O desafio simula um cenário real onde uma aceleradora global busca otimizar seus investimentos identificando startups com maior probabilidade de sucesso.

### Objetivo
Criar um modelo de machine learning que preveja, com alta acurácia, se uma startup será bem-sucedida, apoiando investidores e aceleradoras na tomada de decisões estratégicas.

### Contexto dos Dados
A base contém informações sobre:
- 📈 **Histórico de captação de recursos**
- 🌍 **Localização geográfica**
- 🏭 **Setor de atuação**
- 🔗 **Conexões estratégicas**
- 🏆 **Marcos alcançados**

---

## 📁 Estrutura do Projeto

```
projeto/
│
├── main.py                    # Script principal
├── train.csv                  # Dados de treinamento
├── test.csv                   # Dados de teste
├── sample_submission.csv      # Formato esperado de submissão
└── submission.csv             # Arquivo gerado para submissão
```

---

## 🔧 Requisitos

### Bibliotecas Necessárias

```python
numpy
pandas
scikit-learn
joblib
```

### Instalação

```bash
pip install numpy pandas scikit-learn joblib
```

### ⚠️ Conformidade com Regras da Competição
Este código utiliza **apenas bibliotecas permitidas**:
- ✅ Numpy
- ✅ Pandas
- ✅ Scikit-learn
- ✅ Joblib (parte do scikit-learn para paralelização)

**Métrica de avaliação**: Acurácia (conforme regras da competição)

---

## 📊 Arquivos de Dados

| Arquivo | Descrição |
|---------|-----------|
| `train.csv` | Conjunto de treinamento com features e variável alvo (`labels`) |
| `test.csv` | Conjunto de teste sem a coluna alvo |
| `sample_submission.csv` | Modelo do formato de submissão esperado |
| `submission.csv` | Arquivo final gerado com as predições |

---

## 🧠 Metodologia

### 1. **Engenharia de Features**

O código implementa diversas técnicas de feature engineering:

#### Features Derivadas
- **`age_first_last_diff`**: Diferença entre idade do último e primeiro financiamento
- **`missing_funding_total`**: Indicador binário de valores faltantes em funding
- **`log_funding_total`**: Transformação logarítmica do funding total
- **`funding_per_round`**: Funding médio por rodada de investimento
- **`has_milestones`**: Indicador binário de presença de milestones

#### Features de Interação
- **`funding_milestones_interaction`**: Interação entre funding por rodada e milestones
- **`log_funding_by_age_diff`**: Razão entre log do funding e diferença de idade

#### Indicadores de Missing
Criação de flags para valores ausentes em colunas críticas de idade/tempo.

### 2. **Pré-processamento**

#### Variáveis Numéricas
- Imputação: mediana
- Normalização: StandardScaler

#### Variáveis Categóricas
- Imputação: moda (valor mais frequente)
- Codificação: OneHotEncoder com tratamento de categorias desconhecidas

### 3. **Modelagem**

#### Algoritmo Principal
**HistGradientBoostingClassifier** (Scikit-learn)

Características:
- Eficiente para grandes datasets
- Suporta valores missing nativamente
- Implementação otimizada de Gradient Boosting

#### Validação Cruzada
- **Estratégia**: Stratified K-Fold (5 folds)
- **Objetivo**: Preservar distribuição de classes e evitar overfitting

### 4. **Otimização de Hiperparâmetros**

#### Grid Search Paralelizado
O código implementa busca exaustiva **paralelizada** usando **joblib** nos seguintes hiperparâmetros:

| Hiperparâmetro | Valores Testados |
|----------------|------------------|
| `learning_rate` | [0.03, 0.05, 0.1] |
| `max_depth` | [4, 5, 6] |
| `max_leaf_nodes` | [63, 127] |
| `min_samples_leaf` | [3, 5, 10] |
| `l2_regularization` | [0.0, 0.1] |

**Total de combinações testadas**: 3 × 3 × 2 × 3 × 2 = **108 configurações**

#### Paralelização
- Utiliza `Parallel(n_jobs=-1)` para usar todos os núcleos da CPU
- Acelera drasticamente o grid search
- Cada combinação é avaliada independentemente

### 5. **Otimização de Threshold**

Função implementada **sem scipy** (conforme regras):

```python
def optimize_threshold_acc(y_true, y_probs):
    best_thresh = 0.5
    best_score = -1
    for t in np.linspace(0, 1, 101):
        preds = (y_probs >= t).astype(int)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh
```

- Testa 101 thresholds entre 0 e 1
- Maximiza **acurácia** (métrica da competição)
- Implementação usando apenas numpy e sklearn

---

## ⚙️ Funcionalidades do Código

### Função: `load_data()`
```python
def load_data()
```
- Carrega os três arquivos CSV necessários
- Valida existência dos arquivos
- Retorna DataFrames de treino, teste e sample

### Função: `basic_feature_engineering(df)`
```python
def basic_feature_engineering(df)
```
- Aplica transformações e cria novas features
- Calcula interações entre variáveis
- Gera indicadores de missing values
- Retorna DataFrame enriquecido

### Função: `get_feature_lists(X)`
```python
def get_feature_lists(X)
```
- Identifica colunas numéricas e categóricas
- Retorna duas listas separadas por tipo

### Função: `build_preprocessor(numeric_cols, cat_cols)`
```python
def build_preprocessor(numeric_cols, cat_cols)
```
- Cria pipeline de pré-processamento
- Configura transformadores para cada tipo de variável
- Retorna ColumnTransformer configurado

### Função: `optimize_threshold_acc(y_true, y_probs)`
```python
def optimize_threshold_acc(y_true, y_probs)
```
- Busca threshold que maximiza **acurácia**
- Testa 101 valores linearmente espaçados
- Implementação sem dependências externas
- Retorna valor ótimo de threshold

### Função: `evaluate_combo(params, X, y, preprocessor, skf)`
```python
def evaluate_combo(params, X, y, preprocessor, skf)
```
- Avalia uma combinação de hiperparâmetros
- Realiza validação cruzada completa (5 folds)
- Otimiza threshold para essa combinação
- Retorna score, parâmetros e threshold
- **Projetada para execução paralela**

### Função: `main()`
```python
def main()
```
Orquestra todo o pipeline:
1. Carregamento de dados
2. Feature engineering
3. Pré-processamento
4. Grid search paralelizado com validação cruzada
5. Seleção da melhor configuração
6. Treinamento do modelo final
7. Geração de predições
8. Criação do arquivo de submissão

---

## 🚀 Execução

### Passo a Passo

1. **Prepare os dados**
   ```bash
   # Certifique-se de que os arquivos estão no diretório:
   # - train.csv
   # - test.csv
   # - sample_submission.csv
   ```

2. **Execute o script**

   ```bash
   python main.py
   ```

3. **Acompanhe o progresso**
   - O joblib mostra progresso da paralelização (verbose=10)
   - Tempo de execução: ~5-15 minutos (dependendo do hardware e número de núcleos)
   - Todas as 108 combinações são testadas em paralelo

4. **Submeta os resultados**
   - Arquivo gerado: `submission.csv`
   - Faça upload na plataforma Kaggle

### Exemplo de Saída

```
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    3.2min
[Parallel(n_jobs=-1)]: Done  108 out of 108 | elapsed:    8.5min finished
Melhor configuração encontrada: (0.1, 6, 127, 3, 0.0) | Accuracy=0.8534 | threshold=0.420
Submission salvo em submission.csv | linhas=500 | threshold=0.420
```

---

## 📈 Resultados

### Métrica de Avaliação
**Acurácia**: Proporção de predições corretas sobre o total, conforme especificado nas regras da competição.

### Estratégias Implementadas

✅ **Feature Engineering robusto**
- Features derivadas capturando relações complexas
- Tratamento de missing values
- Transformações logarítmicas

✅ **Validação rigorosa**
- Stratified K-Fold para distribuição balanceada
- Cross-validation para estimativa não enviesada

✅ **Otimização completa e eficiente**
- Grid search em 108 combinações
- **Paralelização** para acelerar busca
- Threshold otimizado para maximizar acurácia

✅ **Pipeline reprodutível**
- Código modular e organizado
- Random seeds fixas para reprodutibilidade
- Apenas bibliotecas permitidas pela competição

✅ **Conformidade com regras**
- Sem uso de scipy ou bibliotecas não autorizadas
- Métrica correta (acurácia)
- Implementações customizadas quando necessário

---
