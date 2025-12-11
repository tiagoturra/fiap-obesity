# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Random seed
SEED = 42
np.random.seed(SEED)

st.set_page_config(
    page_title="Predição de Obesidade - Hospital",
    layout="wide"
)

st.title("🩺 Sistema Preditivo de Obesidade")
st.markdown("""
Este sistema auxilia a equipe médica a **estimar a probabilidade de obesidade** de um paciente,
além de fornecer uma **visão analítica** da base de dados para apoiar ações de prevenção e tratamento.

⚠️ **Aviso importante:** Este modelo é um **auxílio à decisão médica** e não substitui a avaliação clínica.
""")

# =========================
# IMPORTANTE: Definir classes ANTES de carregar os modelos
# =========================

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        self.feature_to_drop = feature_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.feature_to_drop, errors="ignore", inplace=True)
        return X


class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=None):
        self.min_max_scaler = min_max_scaler or [
            'Age', 'Height', 'Weight'
        ]
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        X = X.copy()
        cols = [c for c in self.min_max_scaler if c in X.columns]
        self.scaler.fit(X[cols])
        self.cols_ = cols
        return self

    def transform(self, X):
        X = X.copy()
        if self.scaler is not None:
            X[self.cols_] = self.scaler.transform(X[self.cols_])
        return X


class OneHotEncodingNames(BaseEstimator, TransformerMixin):
  def __init__(self, OneHotEncoding=None):
    # features categóricas nominais
    self.OneHotEncoding = OneHotEncoding or ["Gender", "MTRANS"]

  def fit(self, X, y=None):
    X = X.copy()
    self.encoder = OneHotEncoder(handle_unknown="ignore")
    cols = [c for c in self.OneHotEncoding if c in X.columns]
    self.encoder.fit(X[cols])
    self.cols_ = cols
    self.feature_names_ = self.encoder.get_feature_names_out(self.cols_)
    return self

  def transform(self, X):
    X = X.copy()
    if not hasattr(self, "encoder"):
      raise RuntimeError("OneHotEncodingNames precisa ser ajustado antes de transformar.")

    # Transforma as colunas categóricas
    enc_array = self.encoder.transform(X[self.cols_]).toarray()
    enc_df = pd.DataFrame(enc_array, columns=self.feature_names_, index=X.index)

    # Outras features (que não passaram pelo one-hot)
    outras_features = [c for c in X.columns if c not in self.cols_]

    df_concat = pd.concat([enc_df, X[outras_features]], axis=1)
    return df_concat


class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mappings = {
            "CAEC": ["N/A", "Sometimes", "Frequently", "Always"],
            "CALC": ["Never", "Sometimes", "Frequently", "Always"],
            "faixa_etaria": ["Criança", "Adolescente", "Jovem Adulto", "Adulto", "Idoso"],
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, order in self.mappings.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=order, ordered=True).codes
        return X

class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'is_obese' in df.columns:
            # função smote para super-amostrar a classe minoritária para corrigir os dados desbalanceados
            oversample = SMOTE(sampling_strategy='minority', random_state=SEED)
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'is_obese'], df['is_obese'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("O target não está no DataFrame")
            return df


# =========================
# Funções auxiliares
# =========================

def faixa_etaria(age):
    if age < 12:
        return "Criança"
    elif age < 18:
        return "Adolescente"
    elif age < 30:
        return "Jovem Adulto"
    elif age < 60:
        return "Adulto"
    else:
        return "Idoso"


@st.cache_resource
def load_model():
    """Carrega o modelo e pipeline com tratamento de erros"""
    try:
        model_path = os.path.join('./dest', 'best_model.pkl')
        preprocess_path = os.path.join('./dest', 'preprocess_pipeline.pkl')
        
        if not os.path.exists(model_path):
            st.error(f"❌ Arquivo não encontrado: {model_path}")
            st.info("Certifique-se de que o modelo foi treinado e salvo na pasta './dest/'")
            return None, None
        
        if not os.path.exists(preprocess_path):
            st.error(f"❌ Arquivo não encontrado: {preprocess_path}")
            st.info("Certifique-se de que o pipeline foi treinado e salvo na pasta './dest/'")
            return None, None
        
        model = joblib.load(model_path)
        preprocess = joblib.load(preprocess_path)
        
        st.success("Modelo e Pipeline carregados com sucesso!")
        
        return preprocess, model
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo/pipeline: {str(e)}")
        st.info("""
        **Possíveis soluções:**
        1. Verifique se os arquivos .pkl foram gerados corretamente
        2. Certifique-se de que os modelos foram treinados com as mesmas classes definidas neste arquivo
        3. Tente retreinar o modelo executando o script de treinamento
        """)
        return None, None


@st.cache_data
def load_data():
    df = pd.read_csv("./res/obesity.csv")

    # Mesmos tratamentos do script de modelagem
    df = df.copy()
    df["family_history"] = df["family_history"].map({"yes": 1, "no": 0})
    df["FAVC"] = df["FAVC"].map({"yes": 1, "no": 0})
    df["SMOKE"] = df["SMOKE"].map({"yes": 1, "no": 0})
    df["SCC"] = df["SCC"].map({"yes": 1, "no": 0})

    df["CAEC"] = df["CAEC"].map({"no": "N/A"})
    df["CALC"] = df["CALC"].map({"no": "Never"})
    df["MTRANS"] = df["MTRANS"].str.replace("_", " ", regex=False)
    df["Obesity"] = df["Obesity"].str.replace("_", " ", regex=False)

    df["faixa_etaria"] = df["Age"].apply(faixa_etaria)
    df["is_obese"] = df["Obesity"].isin(
        ["Obesity Type I", "Obesity Type II", "Obesity Type III"]
    ).astype(int)

    return df



preprocess_pipeline, best_model = load_model()
df = load_data()


# =========================
# Menu lateral
# =========================

menu = st.sidebar.radio(
    "Navegação",
    ["🔮 Predição individual", "📊 Painel analítico"]
)

# =========================
# 1) Predição individual
# =========================

if menu == "🔮 Predição individual":
    st.subheader("Predição de obesidade para um paciente")

    with st.form("form_predicao"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gênero", ["Male", "Female"])
            age = st.number_input("Idade (anos)", min_value=1, max_value=120, value=30)
            height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.1)

            family_history = st.selectbox("Histórico familiar de sobrepeso?", ["Sim", "Não"])
            FAVC = st.selectbox("Consumo frequente de alimentos muito calóricos?", ["Sim", "Não"])
            SMOKE = st.selectbox("Fuma?", ["Sim", "Não"])
            SCC = st.selectbox("Controla ingestão calórica?", ["Sim", "Não"])

        with col2:
            FCVC = st.slider("Consumo de vegetais nas refeições [Valores (escala 1–3): 1 raramente, 2 às vezes, 3 sempre]", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
            NCP = st.slider("Número de refeições principais por dia", min_value=1, max_value=8, value=3)
            CAEC = st.selectbox("Come entre as refeições?",
                                ["N/A", "Sometimes", "Frequently", "Always"])
            CH2O = st.slider("Litros de água por dia [Valores (escala 1–3): 1 < 1 L/dia, 2 1–2 L/dia, 3 > 2 L/dia]", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
            FAF = st.slider("Frequência de atividade física [0 nenhuma, 1 ~1–2×/sem, 2 ~3–4×/sem, 3 5×/sem ou mais]", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
            TUE = st.slider("Tempo em dispositivos eletrônicos [0 ~0–2 h/dia, 1 ~3–5 h/dia, 2 > 5 h/dia]", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
            CALC = st.selectbox("Consumo de álcool",
                                ["Never", "Sometimes", "Frequently", "Always"])
            MTRANS = st.selectbox("Meio de transporte principal",
                                  ["Walking", "Bike", "Motorbike",
                                   "Public Transportation", "Automobile"])

        submitted = st.form_submit_button("Calcular predição")

    if submitted:
        # Converte Sim/Não em 1/0
        bin_map = {"Sim": 1, "Não": 0}
        family_history_bin = bin_map[family_history]
        FAVC_bin = bin_map[FAVC]
        SMOKE_bin = bin_map[SMOKE]
        SCC_bin = bin_map[SCC]

        # faixa_etaria derivada da idade
        faixa = faixa_etaria(age)

        # Monta o DataFrame de entrada com as MESMAS features do treino
        input_dict = {
            "Gender": [gender],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "family_history": [family_history_bin],
            "FAVC": [FAVC_bin],
            "FCVC": [FCVC],
            "NCP": [NCP],
            "CAEC": [CAEC],
            "SMOKE": [SMOKE_bin],
            "CH2O": [CH2O],
            "SCC": [SCC_bin],
            "FAF": [FAF],
            "TUE": [TUE],
            "CALC": [CALC],
            "MTRANS": [MTRANS],
            "faixa_etaria": [faixa]
        }

        input_df = pd.DataFrame(input_dict)

        # Pré-processamento + predição
        input_proc = preprocess_pipeline.transform(input_df)
        pred = best_model.predict(input_proc)[0]
        proba = best_model.predict_proba(input_proc)[0, 1]  # prob de ser obeso (classe 1)

        if pred == 1:
            st.error(f"🔴 O modelo indica **ALTA probabilidade de obesidade**")
        else:
            st.success(f"🟢 O modelo indica **baixa probabilidade de obesidade**")

        st.info("""
        Este resultado deve ser interpretado **em conjunto com avaliação clínica, exames físicos e laboratoriais**.
        O modelo não substitui a decisão médica.
        """)

# =========================
# 2) Painel analítico
# =========================

    elif menu == "📊 Painel analítico":
        st.subheader("Painel analítico da base de pacientes")

        powerbi_embed_url = "https://app.powerbi.com/view?r=eyJrIjoiZjllYjYwMGItOTFlNy00Y2FkLTlmZmUtZmU5Yzc4OTNlOGMwIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9" 

        components.iframe(
        powerbi_embed_url, 
        height=700, # Adjust height as needed
        width="100%", # Adjust width as needed
        )

        st.write("This is a Streamlit application displaying a Power BI dashboard.")



   
