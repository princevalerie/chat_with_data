import os
import json
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from openai import OpenAI
import pandasai
from pandasai import SmartDataframe, SmartDatalake, ResponseParser
from pandasai.llm.openai import OpenAI as PandasAI_OpenAI

# Load environment variables
load_dotenv()

# ----------------------------------------------------------
# Helper: Error logging
# ----------------------------------------------------------
def log_error(msg: str):
    if "error_logs" not in st.session_state:
        st.session_state.error_logs = []
    st.session_state.error_logs.append(msg)
    st.warning(msg)

# ----------------------------------------------------------
# Custom Response Parser for Streamlit
# ----------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        st.session_state.answer_cache.append("[Displayed DataFrame]")

    def format_plot(self, result):
        st.image(result["value"])
        st.session_state.answer_cache.append("[Displayed Plot]")

    def format_other(self, result):
        st.write(str(result["value"]))
        st.session_state.answer_cache.append(str(result["value"]))

# ----------------------------------------------------------
# Thinking Agent
# ----------------------------------------------------------
def create_thinking_agent(api_key: str):
    class ThinkingAgent:
        def __init__(self):
            self.client = OpenAI(api_key=api_key)

        def analyze_query(self, query: str, data_info: str) -> str:
            prompt = (
                f"Analyze query: {query}\n"
                f"Data Info: {data_info}\n"
                "Return JSON with keys: understanding, plan, text_explanation_needed,"
                " output_types, visualizations, tables."
            )
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a data analysis expert."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.7,
                    response_format={"type":"json_object"}
                )
                return resp.choices[0].message.content
            except Exception as e:
                log_error(f"ThinkingAgent error: {e}")
                fallback = {
                    "understanding": query,
                    "plan": "Standard analysis",
                    "text_explanation_needed": True,
                    "output_types": ["text","visualization"],
                    "visualizations": [{"description":"Basic plot"}],
                    "tables": []
                }
                return json.dumps(fallback)
    return ThinkingAgent()

# ----------------------------------------------------------
# Visualization Agent (SmartDataframe / SmartDataLake)
# ----------------------------------------------------------
def create_visual_agent(dataframes: dict, api_key: str, use_datalake: bool):
    llm = PandasAI_OpenAI(api_key=api_key, model="gpt-3.5-turbo")
    if use_datalake and len(dataframes) > 1:
        # SmartDataLake across multiple DataFrames
        dfs = list(dataframes.values())
        return SmartDatalake(dfs, config={"llm": llm, "response_parser": StreamlitResponse})
    else:
        # Individual SmartDataframes
        smarts = {}
        for name, df in dataframes.items():
            smarts[name] = SmartDataframe(
                df,
                config={
                    "llm": llm,
                    "response_parser": StreamlitResponse,
                    "enable_cache": False,
                    "save_logs": False,
                    "verbose": False
                }
            )
        return smarts

# ----------------------------------------------------------
# Recommendation Agent
# ----------------------------------------------------------
def create_recommendation_agent(api_key: str):
    class RecommendationAgent:
        def __init__(self):
            self.client = OpenAI(api_key=api_key)

        def generate(self, analysis: str, visuals: str) -> str:
            prompt = (
                f"Based on analysis: {analysis}\n"
                f"And visuals: {visuals}\n"
                "Return JSON with key_insights, recommendations, next_steps."
            )
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a strategy expert."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=0.7,
                    response_format={"type":"json_object"}
                )
                return resp.choices[0].message.content
            except Exception as e:
                log_error(f"RecommendationAgent error: {e}")
                fallback = {
                    "key_insights": ["Insight 1","Insight 2"],
                    "recommendations": ["Reco 1","Reco 2"],
                    "next_steps": ["Step A","Step B"]
                }
                return json.dumps(fallback)
    return RecommendationAgent()

# ----------------------------------------------------------
# File Upload Handler
# ----------------------------------------------------------
def process_file(uploaded) -> pd.DataFrame:
    try:
        ext = uploaded.name.rsplit('.',1)[-1].lower()
        if ext == 'csv':
            return pd.read_csv(uploaded)
        elif ext in ['xlsx','xls']:
            return pd.read_excel(uploaded)
        else:
            raise ValueError(f"Format tidak didukung: {ext}")
    except Exception as e:
        log_error(f"Error upload file: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------
# Main Streamlit App
# ----------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Data Explorer", layout="wide")
    st.title("🔍 AI Data Explorer")

    # Initialize session_state
    for key, default in {
        "dataframes":{}, "answer_cache":[], "error_logs":[], 
        "thinking_result":None, "analysis_mode":"Advanced Analysis",
        "rec_agent":False, "use_datalake":False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.analysis_mode = st.radio(
            "Mode Analisis", ["Basic Analysis","Advanced Analysis"],
            index=1 if st.session_state.analysis_mode=="Advanced Analysis" else 0
        )
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY",""))
        
        st.header("📊 Upload Data")
        up = st.file_uploader("File .csv/.xlsx", type=['csv','xlsx','xls'])
        if up:
            df = process_file(up)
            if not df.empty:
                nm = Path(up.name).stem
                st.session_state.dataframes[nm] = df
                st.success(f"{up.name} berhasil diupload")
                st.dataframe(df.head())

        st.header("🔧 Agent Controls")
        st.session_state.rec_agent = st.checkbox("Enable Recommendation Agent", value=st.session_state.rec_agent)
        st.session_state.use_datalake = st.checkbox("Gunakan SmartDataLake", value=st.session_state.use_datalake)

        if st.session_state.error_logs:
            st.header("⚠️ Errors")
            for e in st.session_state.error_logs:
                st.error(e)

    # No data uploaded
    if not st.session_state.dataframes:
        st.info("Upload file untuk memulai.")
        return

    # Query input
    st.header("💬 Tanyakan Data Anda")
    query = st.text_area("Masukkan pertanyaan:", height=100)
    if st.button("Analisis") and query:
        if not api_key:
            st.error("Masukkan OpenAI API Key!")
            return
        
        st.session_state.answer_cache.clear()

        # Basic Analysis
        if st.session_state.analysis_mode == "Basic Analysis":
            df0 = next(iter(st.session_state.dataframes.values()))
            try:
                lake_or_df = create_visual_agent(st.session_state.dataframes, api_key, st.session_state.use_datalake)
                # single lake object or dict of SmartDataframe
                if isinstance(lake_or_df, dict):
                    lake_or_df[next(iter(lake_or_df))].chat(query)
                else:
                    lake_or_df.chat(query)
            except Exception as e:
                log_error(f"Basic Analysis error: {e}")

        # Advanced Analysis
        else:
            # 1. Thinking
            think = create_thinking_agent(api_key)
            info = json.dumps({
                name:{"columns":list(df.columns),"rows":len(df)}
                for name,df in st.session_state.dataframes.items()
            })
            tr = think.analyze_query(query, info)
            st.session_state.thinking_result = json.loads(tr)
            with st.expander("Hasil Analisis Konteks",expanded=True):
                st.json(st.session_state.thinking_result)

            # 2. Visualization / Lake
            lake_or_df = create_visual_agent(
                st.session_state.dataframes, api_key, st.session_state.use_datalake
            )
            # Jika SmartDataLake
            if not isinstance(lake_or_df, dict):
                st.subheader("Analisis SmartDataLake")
                lake_or_df.chat(query)
            else:
                for name, smart_df in lake_or_df.items():
                    st.subheader(f"Analisis untuk: {name}")
                    for t in st.session_state.thinking_result.get("output_types", []):
                        if t=="text":
                            smart_df.chat(f"Jelaskan: {query}")
                        if t=="table" and st.session_state.thinking_result.get("tables"):
                            desc = ", ".join(tbl["description"] for tbl in st.session_state.thinking_result["tables"])
                            smart_df.chat(f"Buat tabel: {desc}")
                        if t=="visualization" and st.session_state.thinking_result.get("visualizations"):
                            desc = ", ".join(viz["description"] for viz in st.session_state.thinking_result["visualizations"])
                            smart_df.chat(f"Visualisasikan: {desc}")

        # 3. Recommendation
        if st.session_state.rec_agent:
            rec = create_recommendation_agent(api_key)
            out = rec.generate(
                json.dumps(st.session_state.thinking_result),
                json.dumps(st.session_state.answer_cache)
            )
            rd = json.loads(out)
            st.subheader("🚀 Rekomendasi Strategi")
            cols = st.columns(2)
            with cols[0]:
                st.write("**Wawasan Kunci:**")
                [st.write(f"- {i}") for i in rd.get("key_insights",[])]
            with cols[1]:
                st.write("**Rekomendasi:**")
                [st.write(f"- {r}") for r in rd.get("recommendations",[])]
            st.write("**Langkah Selanjutnya:**")
            [st.write(f"- {s}") for s in rd.get("next_steps",[])]

if __name__ == "__main__":
    main()
