# 📊 Previsão de Churn em E-commerce

Este projeto prevê o risco de churn de clientes de e-commerce a partir de dados de comportamento e perfil.

## 🚀 Tecnologias
- Python 3.11+
- pandas, numpy
- scikit-learn
- Streamlit

## 📂 Estrutura
projeto-churn/
│
├── app.py # Aplicação Streamlit
├── requirements.txt # Dependências
├── data/
│ └── Previsao_Churn.csv # Dataset base
└── notebooks/
└── EDA_Modelagem.ipynb # Análises exploratórias (opcional)


## ▶️ Como rodar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/projeto-churn.git
   cd projeto-churn
2. Crie e ative um ambiente virtual (opcional, mas recomendado).
3. Instale as dependências:
pip install -r requirements.txt
4. Rode o app no Streamlit:
streamlit run app.py
5. Abra o navegador em http://localhost:8501.

📈 Funcionalidades

Upload de CSV de clientes.
Previsão do risco de churn com Random Forest.
Exibição da taxa de churn prevista.
Ranking das variáveis mais importantes para o modelo.