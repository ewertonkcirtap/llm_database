import os
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

# ================= CONFIGURAÇÕES =================

# Caminho para a chave de autenticação do Google Cloud
GOOGLE_APPLICATION_CREDENTIALS = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Nome completo da tabela no BigQuery (formato: projeto.dataset.tabela)
NOME_TABELA = "senac-rj.tabelas.dados_fake"

# Inicializa cliente BigQuery
client = bigquery.Client()

# Chave da API da OpenAI
OPENAI_API_KEY = st.secrets["OPENAI"]["key"]

# ================= FUNÇÕES =================

# Variável global para armazenar o último DataFrame consultado
resultado_dataframe = None

import re  # adicionar no topo do seu arquivo

def consultar_bigquery(sql):
    global resultado_dataframe
    try:
        # Remove blocos de código markdown ```sql ... ``` ou ```
        sql_clean = re.sub(r"```(sql)?\n(.+?)\n```", r"\2", sql, flags=re.DOTALL)
        # Remove possíveis backticks isolados no início e fim da string
        sql_clean = sql_clean.strip("` \n")

        query_job = client.query(sql_clean)
        df = query_job.result().to_dataframe()
        resultado_dataframe = df  # salva para uso no Streamlit
        if df.empty:
            return "Nenhum dado encontrado para esta consulta."
        return df.to_markdown(index=False)  # isso o agente ainda pode usar
    except Exception as e:
        return f"Erro ao executar a consulta: {str(e)}"

# Mostra o schema da tabela
def listar_colunas(_):
    return f"""
**Tabela:** `{NOME_TABELA}`  
**Colunas disponíveis:**
- `UF` (STRING): Unidade federativa  
- `VALOR` (FLOAT): Valor da transação  
- `DATA_TRANSACAO` (DATE): Data da transação  
- `TIPO_PAGAMENTO` (STRING): Tipo da transação (ex: Débito, Crédito, Pix)  
- `EMPRESA` (STRING): Nome da Empresa  
- `CNPJ` (STRING) : CNPJ da Empresa
- `MES_ANO` (STRING) : Mes e Ano
"""

# ================= FERRAMENTAS =================

ferramenta_sql = Tool(
    name="Consulta ao BigQuery",
    func=consultar_bigquery,
    description=(
        f"Consulta dados da tabela `{NOME_TABELA}`. "
        "Use SQL para consultar transações, valores por UF, tipo, data, etc. "
        "Colunas disponíveis: UF, VALOR, DATA_TRANSACAO, TIPO_PAGAMENTO, EMPRESA, CNPJ, MES_ANO."
    )
)

ferramenta_schema = Tool(
    name="Schema da Tabela",
    func=listar_colunas,
    description="Mostra as colunas disponíveis da tabela de transações."
)

# ================= AGENTE E EXECUTOR =================

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

agente_base = initialize_agent(
    tools=[ferramenta_schema, ferramenta_sql],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # Ativa logs para facilitar o debug
)

executor = AgentExecutor.from_agent_and_tools(
    agent=agente_base.agent,
    tools=[ferramenta_schema, ferramenta_sql],
    max_iterations=10,           # Aumente se necessário
    max_execution_time=120,      # Limite de tempo em segundos
    verbose=True
)

# ================= INTERFACE STREAMLIT =================

st.set_page_config(page_title="Chatbot with DATABASE", layout="wide")
st.title("🤖 LLM-Powered Chatbot with SQL Access to BigQuery")
st.markdown("Faça perguntas sobre os dados de transações (por UF, tipo pagamento, valor, etc).")

pergunta = st.text_input("Digite sua pergunta:", "")

if pergunta:
    with st.spinner("Consultando..."):
        try:
            resposta = executor.run(pergunta)
        except Exception as e:
            resposta = f"Erro: {str(e)}"

        st.markdown("### Resposta:")
        st.markdown(resposta)

        # Exibe a tabela real, se existir
        if 'resultado_dataframe' in globals() and resultado_dataframe is not None:
            #st.markdown("### Tabela de dados:")
            st.dataframe(resultado_dataframe)
