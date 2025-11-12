import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------------
# Configuração da Página
# -_------------------------------------------------------------------
st.set_page_config(
    page_title="Calculadora de obesidade",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# -------------------------------------------------------------------
# CSS Customizado (Baseado na sua Imagem)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# CSS Customizado (Baseado na sua Imagem)
# -------------------------------------------------------------------
def local_css():
    st.markdown(f"""
    <style>
    /* --- Cor de Fundo Principal --- */
    .stApp {{
        background-color: #000000; /* Preto */
    }}

    /* --- Ocultar Header e Footer do Streamlit --- */
    header[data-testid="stHeader"] {{
        display: none;
        visibility: hidden;
    }}
    footer {{
        display: none;
        visibility: hidden;
    }}

    /* --- Cor do Texto (Labels dos widgets) --- */
    .st-emotion-cache-10trblm, label {{
        color: #FFFFFF !important; /* Branco */
    }}

    /* --- Cor Rosa para Títulos (H1, H2, H3) --- */
    h1, h2, h3 {{
        color: #E6007E !important; /* Rosa/Magenta FIAP */
    }}

    /* --- Estilo do Botão Principal --- */
    .stButton > button {{
        background-color: #E6007E;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
    }}
    .stButton > button:hover {{ background-color: #C0006A; }}
    .stButton > button:active {{ background-color: #A0005A; }}

    /* --- Estilo dos Widgets de Input (Caixas cinzas) --- */
    div[data-baseweb="select"] > div:first-child, /* Selectbox */
    div[data-baseweb="input"] > div:first-child, /* NumberInput */
    div[data-baseweb="text-input"] > div:first-child /* TextInput (para o "Nome") */
    {{
        background-color: #333333 !important;
        border-radius: 5px !important;
        border: 1px solid #555555 !important;
    }}

    /* --- INÍCIO DA CORREÇÃO (PLACEHOLDERS) --- */

    /* 1. Cor do texto JÁ DIGITADO (em Nome, Idade) */
    input {{
        color: #FFFFFF !important;
    }}

    /* 2. Cor do PLACEHOLDER (em Nome, Idade) */
    input::placeholder {{
        color: #FFFFFF !important;
        opacity: 1 !important; 
    }}

    /* 3. Cor do texto e do PLACEHOLDER (em Selectbox) */
    /* Isso afeta "Selecione uma opção" E o valor já selecionado */
    div[data-baseweb="select"] > div:first-child div {{
        color: #FFFFFF !important;
    }}
    
    </style>
    """, unsafe_allow_html=True)

# --- ADICIONE AS DEFINIÇÕES DAS CLASSES CUSTOMIZADAS AQUI ---

class MtransGrouper(BaseEstimator, TransformerMixin):
    """Agrupa as categorias raras de 'mtrans'."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.asarray(X)
        X_series = pd.Series(X_array.flatten(), name='mtrans')
        mtrans_agrupado = X_series.replace(
            ['moto', 'bicicleta', 'caminhando'], 'outros'
        )
        return mtrans_agrupado.values.reshape(-1, 1)


class CalcGrouper(BaseEstimator, TransformerMixin):
    """Agrupa as categorias raras de 'calc'."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.asarray(X)
        X_series = pd.Series(X_array.flatten(), name='calc')
        sempre_freq = X_series.replace(
            'sempre', 'frequentemente'
        )
        return sempre_freq.values.reshape(-1, 1)

class RoundingTransformer(BaseEstimator, TransformerMixin):
    """Arredonda os dados sintéticos (ex: 2.45 -> 2)"""
    def fit(self, X, y=None):
        return self

    def transform(self, X, **kwargs):
        return np.round(X).astype(int)

# -------------------------------------------------------------------
# Carregar os Artefatos
# -------------------------------------------------------------------

@st.cache_resource
def carregar_artefatos():
    """Carrega o pipeline completo e o LabelEncoder."""
    try:
        # As classes customizadas precisam ser definidas ANTES de carregar o pipeline
        pipeline = joblib.load('pipeline_obesidade_completo_rf.joblib')
        label_encoder = joblib.load('label_encoder_rf.joblib')
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error("Erro: Um ou mais arquivos de artefatos não foram encontrados. Certifique-se de que 'pipeline_obesidade_completo_rf.joblib' e 'label_encoder_rf.joblib' estão no diretório correto.")
        return None, None

pipeline, le = carregar_artefatos()

# -------------------------------------------------------------------
# Interface do Usuário (UI) - Layout Principal
# -------------------------------------------------------------------

# Aplicar o CSS customizado
local_css()

# --- Header: Logo e Título ---

col_logo, col_titulo = st.columns([3, 5])

with col_logo:

    st.image("assets/logo_fiap.png", width=100) # <-- Caminho para a imagem

with col_titulo:

    st.markdown("<h1 style='color: #E6007E; text-align: left; margin-top: -10px;'>Calculadora de obesidade</h1>", unsafe_allow_html=True)
    st.markdown("---")

st.markdown("---") # A linha horizontal fica fora das colunas
# Dicionário para coletar os inputs
# As chaves DEVEM ser os nomes exatos das colunas do X_train
inputs_usuario = {}

# Dicionário para mapear valores internos (pipeline) para valores de exibição (Streamlit)
MAPA_TRADUCOES_DISPLAY = {
    'genero': {
        'feminino': 'Feminino',
        'masculino': 'Masculino',
    },
    'sim_nao': {
        'sim': 'Sim',
        'nao': 'Não',
    },
    'mtrans': {
        'carro': 'Automóvel',
        'transporte_publico': 'Transporte Público',
        'caminhando': 'Caminhando',
        'moto': 'Moto',
        'bicicleta': 'Bicicleta',
    },
    'frequencia': {
        'nunca': 'Nunca',
        'as_vezes': 'Às vezes',
        'frequentemente': 'Frequentemente',
        'sempre': 'Sempre',
    },
    'fcvc': {
        0: 'Nunca',
        1: 'Às vezes',
        2: 'Sempre',
    },
    'ncp': {
        0: 'Entre 1 e 2 refeições',
        1: '3 refeições',
        2: 'Mais de 3 refeições',
    },
    'ch20': {
        1: 'Menos que 1 litro',
        2: '1 a 2 litros',
        3: 'Mais que 2 litros',
    },
    'faf': {
        0: 'Nunca',
        1: '1 a 2 dias por semana',
        2: '2 a 4 dias por semana',
        3: '4 ou 5 dias por semana',
    },
    'tue': {
        0: '0-2 horas',
        1: '3-5 horas',
        2: 'Mais que 5 horas',
    },
}

# Listas de opções (os valores *que o pipeline espera*)
opcoes_genero = ['feminino', 'masculino']
opcoes_sim_nao = ['sim', 'nao']
opcoes_frequencia = ['nunca', 'as_vezes', 'frequentemente', 'sempre'] # Para caec, calc
opcoes_mtrans = ['carro', 'transporte_publico', 'caminhando', 'moto', 'bicicleta']
opcoes_fcvc = [0, 1, 2]
opcoes_ncp = [0, 1, 2]
opcoes_ch20 = [1, 2, 3]
opcoes_faf = [0, 1, 2, 3]
opcoes_tue = [0, 1, 2]


# -------------------------------------------------------------------
# Lógica Principal: Previsão e Exibição
# -------------------------------------------------------------------

if pipeline and le:
    
    # --- Seção de Inputs (Layout de Colunas) ---
       
    col3, col4, col5 = st.columns(3)
    
    with col3:

        st.subheader("Dados Pessoais")

        nome_usuario = st.text_input(
            label="Nome", # Label da Imagem 
            value=None,
            placeholder="Insira seu nome",
            max_chars=50
        )
    
        inputs_usuario['idade'] = st.number_input(
            label="Idade", # Label da Imagem
            min_value=1,
            max_value=100,
            value=None,
            placeholder="Insira um valor"
        )
        
        inputs_usuario['genero'] = st.selectbox(
            label="Gênero", # Label da Imagem
            options=opcoes_genero,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['genero'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )

        inputs_usuario['historico_familiar'] = st.selectbox(
            label="Possui histórico familiar de sobrepeso?",
            options=opcoes_sim_nao,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['sim_nao'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
    
    with col4:

        st.subheader("Rotina Pessoal")

        inputs_usuario['faf'] = st.selectbox(
            "Com qual frequência você pratica atividade física?",
            options=opcoes_faf,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['faf'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
        inputs_usuario['mtrans'] = st.selectbox(
            "Qual é o seu meio de transporte principal?",
            options=opcoes_mtrans,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['mtrans'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )

        inputs_usuario['scc'] = st.selectbox(
            label="Você monitora o seu consumo de calorias?",
            options=opcoes_sim_nao,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['sim_nao'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )

        inputs_usuario['tue'] = st.selectbox(
            "Tempo de uso de dispositivos tecnológicos?",
            options=opcoes_tue,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['tue'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
    
    with col5:

        st.subheader("Hábitos Alimentares")

        inputs_usuario['favc'] = st.selectbox(
            label="Você consome alimentos de alta caloria?",
            options=opcoes_sim_nao,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['sim_nao'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )

        inputs_usuario['fcvc'] = st.selectbox(
            label="Com que frequência você consome vegetais?",
            options=opcoes_fcvc,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['fcvc'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
        inputs_usuario['caec'] = st.selectbox(
            label="Você ingere comida entre refeições?",
            options=opcoes_frequencia,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['frequencia'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
        inputs_usuario['ch20'] = st.selectbox(
            "Qual é o seu consumo diário de água?",
            options=opcoes_ch20,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['ch20'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
        
        inputs_usuario['ncp'] = st.selectbox(
            label="Quantas refeições principais você faz por dia?",
            options=opcoes_ncp,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['ncp'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )
        inputs_usuario['calc'] = st.selectbox(
            label="Você consome álcool?",
            options=opcoes_frequencia,
            format_func=lambda x: MAPA_TRADUCOES_DISPLAY['frequencia'].get(x, x),
            index=None,
            placeholder="Selecione uma opção",
        )


    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
    
 # --- Botão e Lógica de Previsão ---
    
    # Criar uma coluna central para o botão, para que não ocupe a tela inteira
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])
    
    with col_btn_2:
        # Botão para executar a previsão
        if st.button("Calcular", use_container_width=True):
            
            # Verificar se todas as entradas foram selecionadas
            if None in inputs_usuario.values():
                st.warning("Por favor, preencha todas as opções antes de calcular.")
            else:
                # 1. Criar um DataFrame com os dados do usuário
                df_input = pd.DataFrame([inputs_usuario])

                try:
                    # 2. Fazer a previsão
                    previsao_numerica = pipeline.predict(df_input)
                    probabilidade = pipeline.predict_proba(df_input)

                    # 3. Mapear o resultado (usando o LabelEncoder)
                    resultado_classe = le.inverse_transform(previsao_numerica)

                    # 4. Exibir o resultado
                    st.markdown("---")
                    st.subheader(f"Resultado da Previsão:")
                    
                    # Centralizar o resultado principal
                    st.markdown(f"""
                        <div style="background-color: #333; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: #E6007E;">{resultado_classe[0]}</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    # Exibir probabilidades
                    st.subheader("Probabilidades por Classe:")
                    nomes_classes = le.classes_
                    df_prob = pd.DataFrame(
                        probabilidade,
                        columns=nomes_classes
                    ).T
                    df_prob.columns = ['Probabilidade']
                    df_prob = df_prob.sort_values(by='Probabilidade', ascending=False)
                    st.dataframe(df_prob.style.format('{:.2%}'), use_container_width=True)

                except Exception as e:
                    st.error(f"Erro ao fazer a previsão: {e}")
else:
    st.warning("Artefatos (pipeline/encoder) não carregados. Não é possível fazer previsões.")
        
    
