# Funcionalidade Cost Weight

## Visão Geral

A funcionalidade `cost_weight` permite ajustar o comportamento dos modelos de classificação para dar prioridade a custo ou latência. Esta funcionalidade é implementada através do ajuste dinâmico de thresholds para modelos que usam `predict_proba`.

## Como Funciona

### Parâmetro Cost Weight

O `cost_weight` é um valor entre 0.0 e 1.0 que determina o balanceamento entre custo e latência:

- **Valores baixos (0.0 - 0.3)**: Favorecem latência, são mais agressivos em classificar objetos como HOT
- **Valores médios (0.4 - 0.6)**: Balanceados entre custo e latência
- **Valores altos (0.7 - 1.0)**: Favorecem custo, são mais conservadores em classificar objetos como HOT

### Ajuste de Thresholds

Para modelos que usam `predict_proba` (como Random Forest, SVM, KNN, etc.), o sistema aplica thresholds dinâmicos:

```python
# Threshold base para HOT
base_hot_threshold = 0.4

# Ajuste baseado no cost_weight
hot_threshold = base_hot_threshold + (cost_weight - 0.5) * 0.8
warm_threshold = 0.3 + (cost_weight - 0.5) * 0.4
```

**Exemplos de thresholds:**
- `cost_weight = 0.2` (agressivo): `hot_threshold = 0.24`, `warm_threshold = 0.18`
- `cost_weight = 0.5` (neutro): `hot_threshold = 0.40`, `warm_threshold = 0.30`
- `cost_weight = 0.8` (conservador): `hot_threshold = 0.56`, `warm_threshold = 0.42`

## Configuração

### Arquivo de Configuração

No arquivo `config/config.yaml`, você pode especificar um valor único ou uma lista de valores:

```yaml
# Valor único
cost_weight: 0.5

# Múltiplos valores
cost_weight: [0.2, 0.5, 0.8]
```

### Exemplos de Configuração

```yaml
# Teste com valores extremos
cost_weight: [0.1, 0.3, 0.5, 0.7, 0.9]

# Foco em latência
cost_weight: [0.1, 0.2, 0.3]

# Foco em custo
cost_weight: [0.7, 0.8, 0.9]
```

## Modelos Afetados

A funcionalidade de threshold se aplica aos seguintes modelos:
- **SVMR** (SVM com RBF kernel)
- **KNN** (K-Nearest Neighbors)
- **RF** (Random Forest)
- **DCT** (Decision Tree)
- **LR** (Logistic Regression)
- **GBC** (Gradient Boosting)
- **ABC** (AdaBoost)
- **ETC** (Extra Trees)
- **MLP** (Neural Network)
- **XGB** (XGBoost)
- **LGBM** (LightGBM)

### Modelos Não Afetados

Os seguintes modelos não usam thresholds (comportamento padrão):
- **HV** (Histogram Voting)
- **AHL** (Always Hot)
- **AWL** (Always Warm)
- **ACL** (Always Cold)
- **ONL** (Online) - usa lógica própria baseada em custos

## Execução

Para executar com diferentes valores de `cost_weight`:

```bash
python main.py --config config/config.yaml
```

O sistema irá:
1. Processar cada valor de `cost_weight` separadamente
2. Aplicar os thresholds apropriados para cada modelo
3. Gerar resultados separados para cada configuração
4. Salvar todos os resultados no arquivo CSV final

## Interpretação dos Resultados

No arquivo de resultados (`resultados_finais.csv`), você verá:

- **model_name**: Nome do modelo
- **cost_weight**: Valor do cost_weight usado
- **model_cost**: Custo total do modelo
- **model_latency**: Latência total do modelo
- **oracle_cost**: Custo do Oracle (melhor possível)
- **oracle_latency**: Latência do Oracle

### Análise de Trade-off

Compare os resultados para diferentes valores de `cost_weight`:

- **cost_weight baixo**: Menor latência, maior custo
- **cost_weight alto**: Maior latência, menor custo

## Exemplo de Uso

```python
# No código, você pode acessar o cost_weight atual:
user_profile = UserProfile(cost_weight=0.3)  # Agressivo
user_profile = UserProfile(cost_weight=0.7)  # Conservador

# Os thresholds são aplicados automaticamente nos modelos
y_pred = apply_cost_weight_thresholds(y_prob, cost_weight, user_profile)
```

## Logs e Debug

Durante a execução, o sistema mostra:

```
Aplicando thresholds baseados no cost_weight = 0.20:
- Threshold para HOT: 0.240 (cost_weight alto = mais conservador)
- Threshold para WARM: 0.180
- Interpretação: cost_weight = 0.20 (agressivo)
```

Isso ajuda a entender como os thresholds estão sendo aplicados para cada configuração. 