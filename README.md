# Trabalho Final — Data Science

Projeto de classificação binária para prever a satisfação de clientes da Olist com base em dados de pedidos, entrega e pagamento. Foram implementados três modelos (Regressão Logística, Naive Bayes e Gradient Boosting) com validação cruzada estratificada, além de um dashboard interativo para exploração dos dados e simulação de predições.

---

# Como Rodar o Projeto

## 1. Instalar dependências

```bash
pip install -r requirements.txt
```

## 2. Baixar o dataset

Acesse https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, baixe e extraia os CSVs na pasta `data/raw/`.

## 3. Executar o pipeline

```bash
python src/data_ingestion.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/modeling.py
```

## 4. Iniciar o dashboard

```bash
streamlit run dashboard/app.py
```

Acesse em: **http://localhost:8501**

## 5. Gerar o relatório PDF

```bash
python report/gerar_relatorio.py
```

O PDF será salvo em `report/Relatorio_Tecnico.pdf`.
