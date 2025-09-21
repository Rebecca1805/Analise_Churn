# === Imports iniciais ===
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import streamlit as st

# Reprodutibilidade
RANDOM_STATE = 23
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# === Configuração do pré-processamento ===
colunas_categoricas = ["Tipo_Assinatura", "Nivel_Satisfacao", "Cidade", "Genero", "Desconto_Aplicado"]
colunas_numericas = ["Gasto_Total", "Itens_Comprados", "Dias_Sem_Compra", "Idade", "Nota_Media"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), colunas_numericas),

        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), colunas_categoricas)
    ]
)

# === Treinamento do modelo (com dataset base) ===
@st.cache_resource
def treinar_modelo():
    df = pd.read_csv("Data/Previsao_Churn.csv")

    # Renomear colunas
    df.rename(columns={
        "Customer ID": "ID_Cliente",
        "Gender": "Genero",
        "Age": "Idade",
        "City": "Cidade",
        "Membership Type": "Tipo_Assinatura",
        "Total Spend": "Gasto_Total",
        "Items Purchased": "Itens_Comprados",
        "Average Rating": "Nota_Media",
        "Discount Applied": "Desconto_Aplicado",
        "Days Since Last Purchase": "Dias_Sem_Compra",
        "Satisfaction Level": "Nivel_Satisfacao"
    }, inplace=True)

    # Criar variável alvo
    df["Churn"] = (df["Dias_Sem_Compra"] > 30).astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    rf_model.fit(X_train, y_train)

    return rf_model

rf_model = treinar_modelo()

# === Interface Streamlit ===
st.set_page_config(page_title="Previsão de Churn", layout="wide")
st.title("📊 Previsão de Churn em E-commerce")
st.write("Faça upload de um CSV de clientes para prever risco de churn.")

st.markdown("""
**📌 Instruções para o upload:**
- O arquivo CSV deve conter as seguintes colunas:
  - `ID_Cliente`
  - `Genero`
  - `Idade`
  - `Cidade`
  - `Tipo_Assinatura`
  - `Gasto_Total`
  - `Itens_Comprados`
  - `Nota_Media`
  - `Desconto_Aplicado`
  - `Dias_Sem_Compra`
  - `Nivel_Satisfacao`
- O app usará essas colunas para prever o risco de churn.
""")

colunas_obrigatorias = [
    "ID_Cliente", "Genero", "Idade", "Cidade", "Tipo_Assinatura",
    "Gasto_Total", "Itens_Comprados", "Nota_Media",
    "Desconto_Aplicado", "Dias_Sem_Compra", "Nivel_Satisfacao"
]

uploaded_file = st.file_uploader("Carregue o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    # Padronizar nomes de colunas
    df_input.rename(columns={
        "Customer ID": "ID_Cliente",
        "Gender": "Genero",
        "Age": "Idade",
        "City": "Cidade",
        "Membership Type": "Tipo_Assinatura",
        "Total Spend": "Gasto_Total",
        "Items Purchased": "Itens_Comprados",
        "Average Rating": "Nota_Media",
        "Discount Applied": "Desconto_Aplicado",
        "Days Since Last Purchase": "Dias_Sem_Compra",
        "Satisfaction Level": "Nivel_Satisfacao"
    }, inplace=True)

    # Verificar colunas obrigatórias
    faltando = [c for c in colunas_obrigatorias if c not in df_input.columns]
    if faltando:
        st.error(f"O arquivo está faltando as colunas: {faltando}")
    else:
        # Só exibe a tabela, sem texto adicional
        st.dataframe(df_input.head())

        # Predições
        preds = rf_model.predict(df_input)
        probs = rf_model.predict_proba(df_input)[:, 1]

        df_input["Pred_Churn"] = preds
        df_input["Prob_Churn"] = probs

        st.subheader("🔮 Resultados das Predições")
        st.dataframe(df_input[["ID_Cliente", "Pred_Churn", "Prob_Churn"]].head(20))

        churn_rate = df_input["Pred_Churn"].mean() * 100
        st.metric("Taxa de Churn Prevista", f"{churn_rate:.2f}%")

        # Importância das variáveis
        importancias = rf_model.named_steps["classifier"].feature_importances_
        feature_names_num = colunas_numericas
        feature_names_cat = rf_model.named_steps["preprocessor"].transformers_[1][1]\
            .named_steps["onehot"].get_feature_names_out(colunas_categoricas)
        feature_names = list(feature_names_num) + list(feature_names_cat)

        importancias_df = pd.DataFrame({
            "Variavel": feature_names,
            "Importancia": importancias
        }).sort_values(by="Importancia", ascending=False).head(10)

        st.write("### 🔑 Top Variáveis Mais Importantes")
        st.bar_chart(importancias_df.set_index("Variavel"))

