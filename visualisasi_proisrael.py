import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import ast
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
from collections import Counter
import io
import hashlib 

# Import SMOTE dari imblearn
from imblearn.over_sampling import SMOTE


# Coba unduh data NLTK yang dibutuhkan
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

# Pastikan stopwords bahasa Indonesia tersedia
try:
    stop_words_id = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_id = set(stopwords.words('indonesian'))


# Atur konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Dashboard Analisis Sentimen")

# --- Fungsi untuk WordCloud per Sentimen ---
@st.cache_data
def generate_sentiment_wordclouds(df, text_column='teks', sentiment_column='label'):
    if text_column not in df.columns or sentiment_column not in df.columns:
        st.error(f"Kolom '{text_column}' atau '{sentiment_column}' tidak ditemukan.")
        return {}, "", ""

    df_sentiment = df.copy()
    df_sentiment['processed_text_for_wc'] = df_sentiment[text_column].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else (str(x) if not pd.isna(x) else "")
    )
    df_sentiment = df_sentiment.dropna(subset=[sentiment_column])

    label_mapping = {0: 'Negatif', 1: 'Positif'}
    df_sentiment[sentiment_column] = df_sentiment[sentiment_column].map(label_mapping)

    positive_texts = " ".join(df_sentiment[df_sentiment[sentiment_column] == 'Positif']['processed_text_for_wc'].tolist())
    negative_texts = " ".join(df_sentiment[df_sentiment[sentiment_column] == 'Negatif']['processed_text_for_wc'].tolist())

    wordclouds = {}
    if positive_texts:
        wordclouds['Positif'] = WordCloud(
            width=800, height=400, background_color='white', colormap='Blues',
            min_font_size=10, collocations=False, stopwords=stop_words_id
        ).generate(positive_texts)

    if negative_texts:
        wordclouds['Negatif'] = WordCloud(
            width=800, height=400, background_color='white', colormap='Reds',
            min_font_size=10, collocations=False, stopwords=stop_words_id
        ).generate(negative_texts)

    return wordclouds, positive_texts, negative_texts

# --- Fungsi untuk menghitung kata paling populer ---
@st.cache_data
def get_most_common_words(texts_string, num_words=10):
    if not texts_string:
        return pd.DataFrame({'Kata': [], 'Jumlah': []})
    
    words = re.findall(r'\b\w+\b', texts_string.lower())
    filtered_words = [word for word in words if word not in stop_words_id and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    most_common_df = pd.DataFrame(word_counts.most_common(num_words), columns=['Kata', 'Jumlah'])
    return most_common_df


# --- Fungsi untuk mendapatkan hash dari file (untuk Caching yang benar) ---
@st.cache_data
def get_file_hash(file_path):
    """Menghitung hash SHA256 dari konten file untuk melacak perubahan."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return np.random.rand()

# --- Fungsi Pemuatan dan Pembagian Data ---
@st.cache_data
def load_and_split_data(file_path, file_hash):
    """Memuat data dan membaginya. Cache akan diperbarui jika file_hash berubah."""
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        st.stop()

    try:
        df_data = pd.read_csv(file_path)

        if 'teks' in df_data.columns:
            def process_text_entry_for_load(x):
                if pd.isna(x): return []
                if isinstance(x, str):
                    x_stripped = x.strip()
                    if x_stripped.startswith('[') and x_stripped.endswith(']'):
                        try:
                            evaluated = ast.literal_eval(x_stripped)
                            return list(evaluated) if isinstance(evaluated, (list, np.ndarray)) else [str(evaluated)]
                        except (ValueError, SyntaxError): pass
                    return [x]
                return x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [str(x)])
            # Kolom 'teks' diproses menjadi list
            df_data['teks'] = df_data['teks'].apply(process_text_entry_for_load)

        embedding_cols = [col for col in df_data.columns if col.startswith('embedding_')]
        if not embedding_cols:
            st.error("Error: Kolom embeddings ('embedding_') tidak ditemukan.")
            st.stop()
        if 'label' not in df_data.columns:
            st.error("Error: Kolom 'label' tidak ditemukan.")
            st.stop()

        for col in embedding_cols:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_data.dropna(subset=embedding_cols + ['label'], inplace=True)

        if len(df_data) == 0:
            st.error("Tidak ada data yang tersisa setelah pembersihan.")
            st.stop()

        X_embeddings = df_data[embedding_cols].values.astype(np.float64)
        y_labels = df_data['label'].values.astype(int)

        test_size = 0.2
        random_seed = 42

        X_train, X_test, y_train, y_test = train_test_split(
            X_embeddings, y_labels, test_size=test_size, random_state=random_seed, stratify=y_labels
        )
        
        train_idx, test_idx, _, _ = train_test_split(
            df_data.index, df_data['label'], test_size=test_size, random_state=random_seed, stratify=df_data['label']
        )
        
        df_train_display = df_data.loc[train_idx]
        df_test_display = df_data.loc[test_idx]

        return df_data, X_train, X_test, y_train, y_test, y_labels, df_train_display, df_test_display

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau membagi data: {e}")
        st.exception(e)
        st.stop()

# --- Fungsi SMOTE ---
@st.cache_data
def apply_smote(X_train_data, y_train_data):
    try:
        smote = SMOTE(random_state=42)
        X_train_smoted, y_train_smoted = smote.fit_resample(X_train_data, y_train_data)
        return X_train_smoted, y_train_smoted
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menerapkan SMOTE: {e}")
        st.exception(e)
        st.stop()

# ==============================================================================
# --- MAIN APP LOGIC ---
# ==============================================================================

# --- Sidebar ---
st.sidebar.header("Konfigurasi Dataset")
default_file_path = "boikot_labeled_and_embedded.csv"
input_combined_file = st.sidebar.text_input("Path File Data CSV", value=default_file_path)

# --- Proses Data dan Model ---
file_hash = get_file_hash(input_combined_file)
df_data, X_train, X_test, y_train, y_test, y_labels_original, df_train_display, df_test_display = load_and_split_data(input_combined_file, file_hash)
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_smote, y_train_smote)
st.session_state['best_model_gnb'] = naive_bayes_model

# --- Struktur Dashboard Utama ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .main-title { font-size: 2.5em; text-align: center; color: #FFFFFF; padding: 20px; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; margin-top: 30px; }
    .stTabs [data-baseweb="tab-list"] button { background-color: #333333; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] button:hover { background-color: #555555; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { background-color: #0BA6DF; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>Klasifikasi Unggahan Terkait Isu Boikot Menggunakan <br> Algoritma <span style='color:#0BA6DF;'>Naive Bayes Classifier</span></h1>",
    unsafe_allow_html=True
)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Data", "Kata Populer & Wordcloud", "Performa Model", "Evaluasi Uji Rinci"])

# ==============================================================================
# --- TAB 1: DISTRIBUSI DATA ---
# ==============================================================================
with tab1:
    st.header("Distribusi Data")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Jumlah Data")
        
        label_counts = pd.Series(y_labels_original).value_counts()
        count_neg = label_counts.get(0, 0)
        count_pos = label_counts.get(1, 0)
        total_data = count_neg + count_pos
        
        summary_data = {
            'No': ['1', '2', ''],
            'Label': ['Negatif', 'Positif', 'Total'],
            'Jumlah Data': [count_neg, count_pos, total_data]
        }
        summary_df = pd.DataFrame(summary_data)
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.subheader("PIE CHART")
        pie_df = pd.DataFrame({
            'Label': ['Negatif', 'Positif'],
            'Jumlah': [count_neg, count_pos]
        })
        fig_pie = px.pie(pie_df, names='Label', values='Jumlah', color_discrete_map={'Negatif':'#2ca02c', 'Positif':'#0BA6DF'}, template="plotly_dark")
        fig_pie.update_traces(textinfo='percent+label', textfont_size=14, marker=dict(line=dict(color='#000000', width=2)))
        fig_pie.update_layout(showlegend=True, title_text='Distribusi Sentimen', title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.subheader("Distribusi Split Train/Test")
        
        total_rows = len(y_labels_original)
        train_count = len(X_train)
        test_count = len(X_test)
        train_percent = (train_count / total_rows) * 100 if total_rows > 0 else 0
        test_percent = (test_count / total_rows) * 100 if total_rows > 0 else 0

        split_df_data = {
            'Keterangan': ['Total Data', 'Data Latih (Train)', 'Data Uji (Test)'],
            'Jumlah': [f'{total_rows}', f'{train_count} ({train_percent:.0f}%)', f'{test_count} ({test_percent:.0f}%)']
        }
        split_df = pd.DataFrame(split_df_data)
        
        st.dataframe(split_df, use_container_width=True, hide_index=True)
        
        st.subheader("Distribusi Label Data Testing")
        
        label_mapping = {0: 'Negatif', 1: 'Positif'}
        y_test_labels = pd.Series(y_test).map(label_mapping)
        test_counts_df = y_test_labels.value_counts().reset_index()
        test_counts_df.columns = ['Label', 'Jumlah']

        fig_test_bar = px.bar(
            test_counts_df,
            x='Label', y='Jumlah', title=f"Distribusi Label Data Testing (Total: {test_count})",
            template="plotly_dark", color='Label',
            color_discrete_map={'Negatif': '#326273', 'Positif': '#5C9653'},
            text_auto=True
        )
        fig_test_bar.update_layout(xaxis_title=None, yaxis_title="Jumlah Data", showlegend=False)
        fig_test_bar.update_traces(textposition='outside', textfont_size=14, textfont_color='white')
        st.plotly_chart(fig_test_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Visualisasi Penyeimbangan Data Latih (SMOTE)")
    col_before, col_after = st.columns(2)

    with col_before:
        train_before_df = pd.DataFrame({'Label': y_train}).replace({0: 'Negatif', 1: 'Positif'})
        fig_before = px.bar(train_before_df['Label'].value_counts(), title='Sebelum SMOTE', template="plotly_dark", text_auto=True)
        st.plotly_chart(fig_before, use_container_width=True, config={'displayModeBar': False})

    with col_after:
        train_after_df = pd.DataFrame({'Label': y_train_smote}).replace({0: 'Negatif', 1: 'Positif'})
        fig_after = px.bar(train_after_df['Label'].value_counts(), title='Sesudah SMOTE', template="plotly_dark", text_auto=True)
        st.plotly_chart(fig_after, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    st.header("Sampel Data Hasil Pembagian")

    tab_train_sample, tab_test_sample = st.tabs(["Data Latih (Train)", "Data Uji (Test)"])

    with tab_train_sample:
        st.write("Berikut adalah seluruh data latih:")
        df_train_display_for_viz = df_train_display.copy()
        df_train_display_for_viz['teks'] = df_train_display_for_viz['teks'].apply(
            lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
        )
        st.dataframe(df_train_display_for_viz, use_container_width=True)

    with tab_test_sample:
        st.write("Berikut adalah seluruh data uji:")
        df_test_display_for_viz = df_test_display.copy()
        df_test_display_for_viz['teks'] = df_test_display_for_viz['teks'].apply(
            lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
        )
        st.dataframe(df_test_display_for_viz, use_container_width=True)

# ==============================================================================
# --- TAB 2: KATA POPULER & WORDCLOUD ---
# ==============================================================================
with tab2:
    st.header("Analisis Teks")
    wordclouds_dict, positive_texts, negative_texts = generate_sentiment_wordclouds(df_data)

    st.markdown("<h2 style='text-align: left; color: white;'>Kata-kata Paling Populer</h2>", unsafe_allow_html=True)
    col_pop_pos, col_pop_neg = st.columns(2)

    with col_pop_pos:
        st.markdown("<h3 style='color: #2ca02c; text-align: left; margin-top: 0;'>Sentimen Positif</h3>", unsafe_allow_html=True)
        most_common_pos_df = get_most_common_words(positive_texts)
        if not most_common_pos_df.empty:
            fig_pos_words = px.bar(
                most_common_pos_df, x='Jumlah', y='Kata', orientation='h',
                title='10 Kata Paling Sering Muncul', template="plotly_dark",
                color='Jumlah', color_continuous_scale=px.colors.sequential.GnBu, text='Jumlah'
            )
            fig_pos_words.update_layout(
                yaxis=dict(
                    categoryorder='total ascending',
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                xaxis_title='Jumlah', yaxis_title=None,
                coloraxis_showscale=False, font=dict(color="white"), title_font_color="white", height=400,
                plot_bgcolor='rgb(30, 30, 30)',
                paper_bgcolor='#0E1117',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            )
            fig_pos_words.update_traces(textposition='outside', textfont=dict(color='white'))
            st.plotly_chart(fig_pos_words, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Tidak ada data kata populer untuk sentimen positif.")

    with col_pop_neg:
        st.markdown("<h3 style='color: #E84A5F; text-align: left; margin-top: 0;'>Sentimen Negatif</h3>", unsafe_allow_html=True)
        most_common_neg_df = get_most_common_words(negative_texts)
        if not most_common_neg_df.empty:
            fig_neg_words = px.bar(
                most_common_neg_df, x='Jumlah', y='Kata', orientation='h',
                title='10 Kata Paling Sering Muncul', template="plotly_dark",
                color='Jumlah', color_continuous_scale=px.colors.sequential.RdPu, text='Jumlah'
            )
            fig_neg_words.update_layout(
                yaxis=dict(
                    categoryorder='total ascending',
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                xaxis_title='Jumlah', yaxis_title=None,
                coloraxis_showscale=False, font=dict(color="white"), title_font_color="white", height=400,
                plot_bgcolor='rgb(30, 30, 30)',
                paper_bgcolor='#0E1117',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            )
            fig_neg_words.update_traces(textposition='outside', textfont=dict(color='white'))
            st.plotly_chart(fig_neg_words, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Tidak ada data kata populer untuk sentimen negatif.")

    st.markdown("---")
    st.subheader("Word Cloud")
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        if 'Positif' in wordclouds_dict:
            st.markdown("<h4 style='text-align: center;'>Sentimen Positif</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordclouds_dict['Positif'], interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Tidak dapat membuat Word Cloud untuk sentimen positif.")
    with col_wc2:
        if 'Negatif' in wordclouds_dict:
            st.markdown("<h4 style='text-align: center;'>Sentimen Negatif</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordclouds_dict['Negatif'], interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Tidak dapat membuat Word Cloud untuk sentimen negatif.")

# ==============================================================================
# --- TAB 3: PERFORMA MODEL ---
# ==============================================================================
with tab3:
    st.markdown("<h2 style='text-align: center; color: white;'>Performa Model Naive Bayes Classifier</h2>", unsafe_allow_html=True)
    
    best_model_gnb = st.session_state['best_model_gnb']
    
    y_pred_gnb_train = best_model_gnb.predict(X_train_smote)
    y_pred_gnb_test = best_model_gnb.predict(X_test)
    
    accuracy_gnb_train = accuracy_score(y_train_smote, y_pred_gnb_train)
    precision_gnb_train = precision_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)
    recall_gnb_train = recall_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)
    f1_gnb_train = f1_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)
    
    accuracy_gnb_test = accuracy_score(y_test, y_pred_gnb_test)
    precision_gnb_test = precision_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)
    recall_gnb_test = recall_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)
    f1_gnb_test = f1_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)

    st.subheader("Tabel Metrik Naive Bayes Classifier")
    
    data_gnb_table = {
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
        'Data Latih (Train)': [accuracy_gnb_train, precision_gnb_train, recall_gnb_train, f1_gnb_train],
        'Data Uji (Test)': [accuracy_gnb_test, precision_gnb_test, recall_gnb_test, f1_gnb_test]
    }
    df_gnb_metrics_table = pd.DataFrame(data_gnb_table)
    st.dataframe(df_gnb_metrics_table.style.format({'Data Latih (Train)': "{:.4f}", 'Data Uji (Test)': "{:.4f}"}), use_container_width=True)
    
    st.subheader("Diagram Performa Naive Bayes Classifier")
    
    metrics_gnb_for_plot = pd.DataFrame({
        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'] * 2,
        'Tipe Data': ['Data Latih (Train)']*4 + ['Data Uji (Test)']*4,
        'Skor': [accuracy_gnb_train, precision_gnb_train, recall_gnb_train, f1_gnb_train,
                 accuracy_gnb_test, precision_gnb_test, recall_gnb_test, f1_gnb_test]
    })

    fig_gnb_metrics = px.bar(
        metrics_gnb_for_plot,
        x='Metrik', y='Skor', color='Tipe Data', barmode='group',
        title='Performa Naive Bayes Classifier', template="plotly_dark", text_auto='.3f'
    )
    fig_gnb_metrics.update_layout(
        xaxis_title=None, yaxis_title="Skor", legend_title_text=None,
        font=dict(color="white"), title_font_color="white",
        yaxis=dict(range=[0.4, 1.0])
    )
    fig_gnb_metrics.update_traces(textposition='outside', textfont=dict(color='white'))
    st.plotly_chart(fig_gnb_metrics, use_container_width=True, config={'displayModeBar': False})

# ==============================================================================
# --- TAB 4: EVALUASI UJI RINCI ---
# ==============================================================================
with tab4:
    st.header("Evaluasi Rinci pada Data Uji")
    best_model_gnb = st.session_state['best_model_gnb']
    y_pred_nb = best_model_gnb.predict(X_test)
    accuracy_gnb_test_tab4 = accuracy_score(y_test, y_pred_nb)

    st.markdown(f"### Naive Bayes Classifier")
    st.markdown(f"**Akurasi pada Data Uji:** `{accuracy_gnb_test_tab4:.4f}`")
    
    col_report, col_matrix = st.columns(2)

    with col_report:
        st.write("##### Laporan Klasifikasi")
        report_dict_nb = classification_report(y_test, y_pred_nb, target_names=['Negatif', 'Positif'], output_dict=True, zero_division=0)
        df_report_nb = pd.DataFrame(report_dict_nb).transpose()
        st.dataframe(df_report_nb.style.format("{:.2f}"), use_container_width=True)

    with col_matrix:
        st.write("##### Matriks Konfusi")
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        fig_nb, ax_nb = plt.subplots(facecolor='#0E1117')
        
        sns.heatmap(
            cm_nb, annot=True, fmt='d', cmap='viridis', cbar=False,
            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=ax_nb,
            annot_kws={"color": "white"}
        )
        ax_nb.set_xlabel('Prediksi', color='white')
        ax_nb.set_ylabel('Aktual', color='white')
        ax_nb.set_title('Matriks Konfusi', color='white')
        
        ax_nb.tick_params(axis='x', colors='white')
        ax_nb.tick_params(axis='y', colors='white')
        
        ax_nb.set_facecolor('#0E1117')

        st.pyplot(fig_nb)

    st.markdown("---")
    st.subheader("Tabel Hasil Prediksi Data Uji")
    
    df_results = df_test_display.copy()
    df_results['teks'] = df_results['teks'].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
    )
    df_results['True Label'] = df_results['label'].map({0: 'Negatif', 1: 'Positif'})
    df_results['Predicted Label (Naive Bayes Classifier)'] = y_pred_nb
    df_results['Predicted Label (Naive Bayes Classifier)'] = df_results['Predicted Label (Naive Bayes Classifier)'].map({0: 'Negatif', 1: 'Positif'})
    st.dataframe(df_results[['teks', 'True Label', 'Predicted Label (Naive Bayes Classifier)']], use_container_width=True)