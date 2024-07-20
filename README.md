# Análise de Previsão de Preços das Ações da Petrobras (PETR4.SA) usando Modelos LSTM

## Informações do Projeto

- **Aluno**: Laisson Bruno dos Reis Germano
- **Professor**: Dr Adriano Alonso Veloso
- **Disciplina**: Aprendizado de Máquina
- **Programa**: UFMG - Programa de pós graduação stricto sensu em ciência da computação - Mestrado

## Introdução

Este projeto tem como objetivo prever os preços das ações da Petrobras (PETR4.SA) utilizando modelos de Redes Neurais Recorrentes (RNN) do tipo LSTM (Long Short-Term Memory). Foram treinados dois modelos com diferentes taxas de aprendizado para avaliar seu desempenho e precisão na previsão dos preços das ações.

## Métodos Utilizados

### Obtenção dos Dados

Os dados históricos dos preços das ações da Petrobras foram obtidos utilizando a biblioteca `yfinance`. O período considerado foi de 01/01/2010 a 31/12/2023.

```python
import yfinance as yf

petr4 = yf.Ticker("PETR4.SA")
df = petr4.history(period="max")
df_filtered = df[(df.index >= "2010-01-01") & (df.index <= "2023-12-31")]
```

### Preparação dos Dados

Os dados foram normalizados utilizando `MinMaxScaler` para garantir que todos os valores estejam em uma faixa entre 0 e 1, facilitando o treinamento do modelo.

```python
from sklearn.preprocessing import MinMaxScaler

data = df_filtered['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

### Criação de Sequências

Para treinar o modelo LSTM, os dados foram organizados em sequências de 60 dias.

```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
```

### Divisão dos Dados

Os dados foram divididos em conjuntos de treino e teste na proporção de 80/20.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Criação e Treinamento dos Modelos

Dois modelos LSTM foram criados e treinados com diferentes taxas de aprendizado (0.001 e 0.01).

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_model(learning_rate):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

learning_rates = [0.001, 0.01]
models = []
histories = []

for lr in learning_rates:
    model = create_model(lr)
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, verbose=1)
    models.append(model)
    histories.append(history)
```

### Avaliação dos Modelos

Os modelos foram avaliados utilizando as métricas MSE, RMSE, MAE e acurácia.

```python
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y))
    return mse, rmse, mae

results = []
for i, model in enumerate(models):
    train_results = evaluate_model(model, X_train, y_train)
    test_results = evaluate_model(model, X_test, y_test)
    results.append((f"Modelo {i+1} (LR: {learning_rates[i]})", train_results, test_results))
```

### Cálculo da Acurácia

A acurácia foi calculada para os conjuntos de treino e teste.

```python
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    
    correct_predictions = np.sum(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero) <= tolerance)
    total_predictions = len(y_true_non_zero)
    
    if total_predictions == 0:
        return 0
    
    accuracy = correct_predictions / total_predictions
    return accuracy

train_accuracy = calculate_accuracy(y_train, models[0].predict(X_train))
test_accuracy = calculate_accuracy(y_test, models[0].predict(X_test))

print(f"Acurácia no treinamento: {train_accuracy * 100:.2f}%")
print(f"Acurácia no teste: {test_accuracy * 100:.2f}%")
```

### Visualização dos Resultados

Os resultados foram visualizados através de gráficos.

```python
import plotly.graph_objects as go

figures = []

fig1 = go.Figure()
for i, history in enumerate(histories):
    fig1.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name=f'Modelo {i+1} - Treino (LR: {learning_rates[i]})'))
    fig1.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name=f'Modelo {i+1} - Validação (LR: {learning_rates[i]})'))
fig1.update_layout(title='Perda do Modelo durante o Treinamento', xaxis_title='Época', yaxis_title='Perda')
figures.append(fig1)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), mode='lines', name='Valores Reais'))
for i, model in enumerate(models):
    fig2.add_trace(go.Scatter(y=scaler.inverse_transform(model.predict(X_test)).flatten(), mode='lines', name=f'Previsões Modelo {i+1}'))
fig2.update_layout(title='Previsões vs Valores Reais', xaxis_title='Tempo', yaxis_title='Preço da Ação')
figures.append(fig2)

metrics = ['MSE', 'RMSE', 'MAE']
fig3 = go.Figure()
for i, (name, train_metrics, test_metrics) in enumerate(results):
    fig3.add_trace(go.Bar(x=metrics, y=train_metrics, name=f'{name} - Treino'))
    fig3.add_trace(go.Bar(x=metrics, y=test_metrics, name=f'{name} - Teste'))
fig3.update_layout(title='Métricas de Erro e Eficiência', xaxis_title='Métricas', yaxis_title='Valor')
figures.append(fig3)

last_sequence = scaled_data[-seq_length:]
next_year_predictions = []

for _ in range(252):
    x_input = last_sequence.reshape((1, seq_length, 1))
    next_price = models[0].predict(x_input)[0]
    next_year_predictions.append(next_price)
    last_sequence = np.append(last_sequence[1:], next_price)

next_year_predictions = scaler.inverse_transform(np.array(next_year_predictions).reshape(-1, 1))

fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=next_year_predictions.flatten(), mode='lines', name='Previsões para o próximo ano'))
fig4.update_layout(title='Previsões das Ações para o Próximo Ano', xaxis_title='Dias', yaxis_title='Preço da Ação')
figures.append(fig4)

for fig in figures:
    fig.show()
```

## Escolha das Métricas

Para avaliar o desempenho dos modelos LSTM, foram utilizadas as seguintes métricas:

1. **Mean Squared Error (MSE):**
   - Definição: Média dos quadrados das diferenças entre os valores previstos e os valores reais.
   - Fórmula: $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - Explicação: Quanto menor o MSE, melhor o desempenho do modelo. O MSE penaliza erros maiores mais severamente devido ao termo quadrático.

2. **Root Mean Squared Error (RMSE):**
   - Definição: Raiz quadrada da média dos quadrados das diferenças entre os valores previstos e os valores reais.
   - Fórmula: $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   - Explicação: O RMSE é mais fácil de interpretar que o MSE, pois está na mesma unidade dos dados originais. Quanto menor o RMSE, melhor o desempenho do modelo.

3. **Mean Absolute Error (MAE):**
   - Definição: Média dos valores absolutos das diferenças entre os valores previstos e os valores reais.
   - Fórmula: $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - Explicação: O MAE é menos sensível a outliers comparado ao MSE e RMSE. Representa o erro médio em termos absolutos. Quanto menor o MAE, melhor o desempenho do modelo.

4. **Acurácia:**
   - Definição: Porcentagem de previsões que estão dentro de uma tolerância de 5% do valor real.
   - Fórmula: $$Acurácia = \frac{\text{Número de previsões dentro da tolerância}}{\text{Número total de previsões}} \times 100\%$$
   - Explicação: Uma medida mais intuitiva da precisão do modelo. Uma acurácia de 90% significa que 90% das previsões estão dentro de 5% do valor real.

## Resultados

Os resultados incluem gráficos de:
1. Perda do Modelo durante o Treinamento
2. Previsões vs Valores Reais
3. Métricas de Erro e Eficiência
4. Previsões para o próximo ano

Além disso, a acurácia dos modelos foi calculada para os conjuntos de treino e teste.

## Escolha do Tema

Escolhi este tema devido à minha experiência pessoal com investimentos no mercado de ações.

## Conclusão

Os modelos LSTM utilizados neste projeto mostraram-se eficazes na previsão dos preços das ações da Petrobras. As métricas de erro, curvas de perda e acurácia sugerem que os modelos generalizam bem sem overfitting significativo. As previsões para o próximo ano fornecem insights sobre as possíveis tendências futuras dos preços das ações.
## Referências

1. Veloso, Adriano Alo. (2024). Statistical Learning and Deep Learning. https://homepages.dcc.ufmg.br/~adrianov/ml/pres.pdf

2. Brownlee, J. (2018). How to Develop LSTM Models for Time Series Forecasting. Machine Learning Mastery. https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

3. Datacamp. (n.d.). Stock Market Predictions with LSTM in Python. https://www.datacamp.com/tutorial/lstm-python-stock-market

4. Towards Data Science. (2020). Predicting Stock Prices Using LSTM. https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
