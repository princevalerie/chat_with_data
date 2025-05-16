import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import Image

import pandasai
from pandasai.llm import Gemini
from pandasai.responses.response_parser import ResponseParser

# Load environment variables from .env file
load_dotenv()

# Fungsi untuk mencatat error ke session state
def log_error(message):
    st.session_state.error_logs.append(message)
    st.warning(message)

# -----------------------------------------------------------------------------
# Custom Response Parser for Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        """Tampilkan dataframe dan simpan pesan placeholder ke cache."""
        st.dataframe(result["value"])
        st.session_state.answer_cache.append("[Displayed DataFrame]")
        return

    def format_plot(self, result):
        """Tampilkan plot dan simpan pesan placeholder ke cache."""
        st.image(result["value"])
        st.session_state.answer_cache.append("[Displayed Plot]")
        return

    def format_other(self, result):
        """Tampilkan hasil lain sebagai teks dan simpan ke cache."""
        st.write(str(result["value"]))
        st.session_state.answer_cache.append(str(result["value"]))
        return

# -----------------------------------------------------------------------------
# Agent 1: Thinking Agent - Analysis Context Understanding
# -----------------------------------------------------------------------------
def create_thinking_agent(api_key):
    """
    Creates a thinking agent that determines analysis type (basic or advanced)
    and provides analysis plan and visualization suggestions
    """
    class ThinkingAgent:
        def __init__(self, api_key):
            self.api_key = api_key
            self.llm = GoogleGemini(api_key=api_key, model="gemini-pro")
        
        def analyze_query(self, query, data_info):
            """Analyze query for data context"""
            try:
                # Ekstrak informasi data
                data_context = json.loads(data_info) if isinstance(data_info, str) else data_info
                
                # Buat rencana analisis dasar terbuka (tanpa membatasi)
                analysis = {
                    "understanding": f"Analisis data berdasarkan query: {query}",
                    "plan": "Rencana analisis fleksibel berdasarkan kebutuhan data",
                    "text_explanation_needed": True,
                    "output_types": ["text", "visualization"],
                    "visualizations": [],
                    "tables": []
                }
                
                # Deteksi kebutuhan visualisasi dari query tanpa membatasi jenis
                visualization_needs = []
                
                # Tambahkan kebutuhan visualisasi berdasarkan konteks query
                if "trend" in query.lower() or "tren" in query.lower():
                    visualization_needs.append({"description": "Visualisasi untuk menunjukkan tren dari waktu ke waktu"})
                
                if "compare" in query.lower() or "bandingkan" in query.lower() or "perbandingan" in query.lower():
                    visualization_needs.append({"description": "Visualisasi untuk membandingkan nilai antar kategori"})
                    
                if "distribution" in query.lower() or "distribusi" in query.lower():
                    visualization_needs.append({"description": "Visualisasi untuk menunjukkan distribusi data"})
                    
                if "relationship" in query.lower() or "hubungan" in query.lower() or "korelasi" in query.lower():
                    visualization_needs.append({"description": "Visualisasi untuk menunjukkan hubungan antar variabel"})
                    
                if "part of whole" in query.lower() or "bagian dari" in query.lower() or "proporsi" in query.lower():
                    visualization_needs.append({"description": "Visualisasi untuk menunjukkan proporsi atau bagian dari keseluruhan"})
                
                # Untuk query umum, tambahkan visualisasi umum
                if not visualization_needs:
                    visualization_needs.append({"description": "Visualisasi yang paling informatif sesuai data"})
                
                # Jika query mengandung kata-kata yang menunjukkan analisis mendalam, tambahkan beberapa visualisasi
                if any(word in query.lower() for word in ["detail", "mendalam", "komprehensif", "lengkap", "advanced", "lanjut", "complex"]):
                    if len(visualization_needs) < 2:
                        visualization_needs.append({"description": "Visualisasi tambahan untuk menunjukkan aspek lain dari data"})
                        visualization_needs.append({"description": "Visualisasi untuk insight tersembunyi dalam data"})
                
                # Deteksi kebutuhan tabel
                table_needs = []
                
                # Jika query membutuhkan numerik atau statistik
                if any(word in query.lower() for word in ["tabel", "table", "angka", "statistik", "rata-rata", "jumlah", "perhitungan", "ringkasan"]):
                    table_needs.append({"description": "Tabel yang menunjukkan data utama dari query"})
                    
                # Jika query menunjukkan kebutuhan perbandingan rinci
                if any(word in query.lower() for word in ["bandingkan", "perbandingan", "detail", "rinci"]):
                    table_needs.append({"description": "Tabel ringkasan untuk perbandingan kategori"})
                    
                # Untuk analisis lanjutan, selalu sediakan tabel ringkasan
                if any(word in query.lower() for word in ["lanjut", "advanced", "mendalam", "komprehensif"]):
                    if not table_needs:
                        table_needs.append({"description": "Tabel ringkasan statistik dari data"})
                
                # Tambahkan semua kebutuhan yang terdeteksi
                analysis["visualizations"] = visualization_needs
                analysis["tables"] = table_needs
                
                # Jika butuh tabel, tambahkan ke output_types
                if table_needs:
                    analysis["output_types"].append("table")
                
                return json.dumps(analysis)
            except Exception as e:
                log_error(f"Error dalam analisis query: {e}")
                return json.dumps({"error": str(e)})
        
    return ThinkingAgent(api_key)

# -----------------------------------------------------------------------------
# Agent 2: Visualization Agent - PandasAI Implementation
# -----------------------------------------------------------------------------
def create_visualization_agent(dataframes, api_key):
    """
    Create visualization agent using PandasAI to process multiple visualization tasks
    """
    if not dataframes:
        return None
    
    try:
        llm = GoogleGemini(api_key=api_key, model="gemini-pro")
        pandas_ai_dfs = []
        
        for name, df in dataframes.items():
            # Create PandasAI SmartDataframe for each dataframe
            smart_df = pandasai.SmartDataframe(
                df, 
                config={
                    "llm": llm,
                    "response_parser": StreamlitResponse,
                    "enable_cache": False,
                    "save_logs": False,
                    "verbose": False
                }
            )
            pandas_ai_dfs.append((name, smart_df))
        
        return pandas_ai_dfs
    except Exception as e:
        log_error(f"Error creating visualization agent: {e}")
        return None

# -----------------------------------------------------------------------------
# Agent 3: Recommendation Agent - Strategy Suggestions
# -----------------------------------------------------------------------------
def create_recommendation_agent(api_key):
    """
    Creates a recommendation agent that provides strategic insights based on visualizations
    """
    class RecommendationAgent:
        def __init__(self, api_key):
            self.api_key = api_key
            self.llm = GoogleGemini(api_key=api_key, model="gemini-pro")
        
        def generate_recommendations(self, analysis_result, visualization_result):
            """Generate recommendations based on analysis and visualization results"""
            try:
                # Untuk implementasi sederhana, langsung buat rekomendasi dasar
                recommendations = {
                    "key_insights": [
                        "Wawasan 1 berdasarkan data",
                        "Wawasan 2 berdasarkan data"
                    ],
                    "recommendations": [
                        "Rekomendasi 1 berdasarkan analisis",
                        "Rekomendasi 2 berdasarkan analisis"
                    ],
                    "next_steps": [
                        "Langkah selanjutnya 1",
                        "Langkah selanjutnya 2"
                    ]
                }
                
                return json.dumps(recommendations)
            except Exception as e:
                log_error(f"Error dalam membuat rekomendasi: {e}")
                return json.dumps({"error": str(e)})
        
    return RecommendationAgent(api_key)

# -----------------------------------------------------------------------------
# File Upload and Processing
# -----------------------------------------------------------------------------
def process_uploaded_file(uploaded_file):
    """Process uploaded Excel or CSV file and convert to DataFrame"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return df
    except Exception as e:
        log_error(f"Error processing file: {e}")
        return None

# -----------------------------------------------------------------------------
# Main function for Streamlit app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Data Explorer", layout="wide")
    st.title("🔍 AI Data Explorer")

    # Inisialisasi session state jika belum ada
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = []
    if "error_logs" not in st.session_state:
        st.session_state.error_logs = []
    if "rec_agent_enabled" not in st.session_state:
        st.session_state.rec_agent_enabled = False
    if "thinking_result" not in st.session_state:
        st.session_state.thinking_result = None
    if "visualization_result" not in st.session_state:
        st.session_state.visualization_result = None
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "Advanced Analysis"

    # Sidebar: API key, file upload, dan control panel
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Pilihan mode analisis
        st.session_state.analysis_mode = st.radio(
            "Mode Analisis",
            ["Basic Analysis", "Advanced Analysis"],
            index=1 if st.session_state.analysis_mode == "Advanced Analysis" else 0
        )
        
        # API Keys (menggunakan Gemini untuk kedua mode)
        gemini_api_key = st.text_input("Gemini API Key", 
                              value=os.getenv("GEMINI_API_KEY", ""),
                              type="password", 
                              key="gemini_api_key")
        
        st.header("📊 Upload Data")
        uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file)
            if df is not None:
                filename = uploaded_file.name.split('.')[0]
                st.session_state.dataframes[filename] = df
                st.success(f"File {uploaded_file.name} berhasil diupload!")
                
                # Tampilkan preview data
                st.subheader(f"Preview: {filename}")
                st.dataframe(df.head(5))
        
        # Toggle untuk recommendation agent
        st.header("🔧 Agent Controls")
        st.session_state.rec_agent_enabled = st.toggle("Aktifkan Recommendation Agent", st.session_state.rec_agent_enabled)
        
        # Tampilkan dataframes yang sudah diupload
        if st.session_state.dataframes:
            st.subheader("📈 Datasets Loaded")
            for name in st.session_state.dataframes.keys():
                st.write(f"- {name} ({len(st.session_state.dataframes[name])} rows)")
        
        # Error log
        if st.session_state.error_logs:
            st.subheader("⚠️ Error Log")
            for err in st.session_state.error_logs:
                st.error(err)

    # Main content
    if st.session_state.dataframes:
        st.header("💬 Tanyakan tentang Data Anda")
        
        # Input query
        with st.form(key="query_form"):
            prompt = st.text_area("Masukkan query analisis Anda:", height=100)
            submitted = st.form_submit_button("Analisis")
            
            if submitted and prompt:
                # Basic Analysis Mode
                if st.session_state.analysis_mode == "Basic Analysis":
                    if not gemini_api_key:
                        st.error("Mohon masukkan Gemini API Key terlebih dahulu!")
                    else:
                        st.session_state.answer_cache.clear()  # Refresh output cache
                        
                        with st.spinner("Menganalisis data Anda..."):
                            try:
                                # Gunakan dataframe pertama jika ada beberapa
                                df_name, df = next(iter(st.session_state.dataframes.items()))
                                
                                # Create SmartDataframe with config
                                smart_df = pandasai.SmartDataframe(
                                    df,
                                    config={
                                        "llm": GoogleGemini(api_key=gemini_api_key, model="gemini-pro"),
                                        "response_parser": StreamlitResponse,
                                        "enable_cache": False,
                                        "save_logs": False,
                                        "verbose": False
                                    }
                                )
                                
                                # Get answer
                                response = smart_df.chat(prompt)
                                
                            except Exception as e:
                                log_error(f"Error dalam Basic Analysis: {e}")
                                st.error(f"Terjadi kesalahan: {e}")
                
                # Advanced Analysis Mode
                else:
                    if not gemini_api_key:
                        st.error("Mohon masukkan Gemini API Key terlebih dahulu!")
                    else:
                        st.session_state.answer_cache.clear()  # Refresh output cache
                        
                        with st.spinner("Menganalisis query Anda..."):
                            # 1. Thinking Agent - Analisis konteks
                            thinking_agent = create_thinking_agent(gemini_api_key)
                            data_info = {name: {"columns": list(df.columns), "rows": len(df)} 
                                        for name, df in st.session_state.dataframes.items()}
                            
                            thinking_result = thinking_agent.analyze_query(prompt, json.dumps(data_info))
                            
                            st.session_state.thinking_result = json.loads(thinking_result)
                            
                            # Tampilkan hasil thinking agent
                            with st.expander("Hasil Analisis Konteks", expanded=True):
                                st.json(thinking_result)
                        
                        with st.spinner("Membuat visualisasi..."):
                            # 2. Visualization Agent - Visualisasi dengan PandasAI
                            visualization_agents = create_visualization_agent(st.session_state.dataframes, gemini_api_key)
                            
                            if visualization_agents:
                                # Ekstrak rencana dari thinking agent
                                thinking_data = st.session_state.thinking_result
                                output_types = thinking_data.get("output_types", ["text", "visualization"])
                                
                                # Proses setiap dataframe dengan PandasAI
                                for name, smart_df in visualization_agents:
                                    st.subheader(f"Analisis untuk: {name}")
                                    
                                    # 1. Selalu buat penjelasan teks terlebih dahulu
                                    if "text" in output_types:
                                        explanation_prompt = f"Berikan penjelasan tekstual komprehensif tentang: {prompt}"
                                        try:
                                            response = smart_df.chat(explanation_prompt)
                                        except Exception as e:
                                            log_error(f"Error dalam membuat penjelasan: {e}")
                                    
                                    # 2. Buat tabel jika diminta
                                    if "table" in output_types and thinking_data.get("tables"):
                                        table_descriptions = [table["description"] for table in thinking_data.get("tables", [])]
                                        combined_table_prompt = f"""Berdasarkan query: '{prompt}'
                                        
                                        Buatkan analisis tabular yang mencakup semua aspek berikut:
                                        - {', '.join(table_descriptions)}
                                        
                                        Tampilkan dalam format tabel yang paling informatif.
                                        """
                                        
                                        try:
                                            response = smart_df.chat(combined_table_prompt)
                                        except Exception as e:
                                            log_error(f"Error dalam membuat tabel: {e}")
                                    
                                    # 3. Buat visualisasi
                                    if "visualization" in output_types:
                                        visualizations = thinking_data.get("visualizations", [])
                                        if visualizations:
                                            viz_descriptions = [viz['description'] for viz in visualizations]
                                            combined_viz_prompt = f"""Menganalisis data berdasarkan query: '{prompt}'
                                            
                                            Buatkan analisis visual yang komprehensif dengan kebutuhan:
                                            - {', '.join(viz_descriptions)}
                                            
                                            Pilih jenis visualisasi yang paling optimal untuk setiap kebutuhan tersebut.
                                            """
                                            
                                            try:
                                                response = smart_df.chat(combined_viz_prompt)
                                            except Exception as e:
                                                log_error(f"Error dalam membuat visualisasi: {e}")
                        
                        # 3. Recommendation Agent - jika diaktifkan
                        if st.session_state.rec_agent_enabled:
                            with st.spinner("Membuat rekomendasi strategi..."):
                                recommendation_agent = create_recommendation_agent(gemini_api_key)
                                
                                recommendations = recommendation_agent.generate_recommendations(
                                    thinking_result,
                                    json.dumps(st.session_state.answer_cache)
                                )
                                
                                # Tampilkan rekomendasi
                                st.subheader("🚀 Rekomendasi Strategi")
                                rec_data = json.loads(recommendations)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Wawasan Kunci:**")
                                    for insight in rec_data.get("key_insights", []):
                                        st.write(f"- {insight}")
                                
                                with col2:
                                    st.write("**Rekomendasi:**")
                                    for rec in rec_data.get("recommendations", []):
                                        st.write(f"- {rec}")
                                
                                st.write("**Langkah Selanjutnya:**")
                                for step in rec_data.get("next_steps", []):
                                    st.write(f"- {step}")
    else:
        st.info("Upload file Excel atau CSV untuk memulai analisis.")

if __name__ == "__main__":
    main()
