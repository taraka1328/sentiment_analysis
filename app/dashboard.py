import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from wordcloud import WordCloud, STOPWORDS
import io
import base64
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# COLOR PALETTE — Green / Yellow / Red theme
# ─────────────────────────────────────────────
PALETTE = {
    'positive': '#5BAD7E',   # muted sage green
    'neutral':  '#C49A3C',   # muted amber
    'negative': '#C0574E',   # muted rose red
}
BG_PALE = {
    'positive': '#F2FAF5',
    'neutral':  '#FDF8EE',
    'negative': '#FDF2F1',
}
BORDER_COLOR = {
    'positive': '#5BAD7E',
    'neutral':  '#C49A3C',
    'negative': '#C0574E',
}

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #EAF3FB;
    color: #1e293b;
}
header[data-testid="stHeader"] { display: none; }
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1200px;
    background-color: #EAF3FB;
}

/* Title */
.dash-title {
    font-size: 1.7rem; font-weight: 700;
    letter-spacing: -0.5px; color: #0f172a; 
    margin-bottom: 2rem; /* Adjusted margin since subtitle is removed */
    text-align: center;
}

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 120px;
    background: #f5f9fd;
    border: 1px solid #d0e4f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.total::before { background: #7b88d4; }
.kpi-card.pos::before   { background: #5BAD7E; }
.kpi-card.neg::before   { background: #C0574E; }
.kpi-card.neu::before   { background: #C49A3C; }

.kpi-label {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.9rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a; line-height: 1;
}
.kpi-pct { font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem; }
.kpi-icon { font-size: 1.3rem; float: right; opacity: 0.18; margin-top: -2rem; }

/* Section headings */
.section-head {
    font-size: 0.76rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #475569;
    margin: 1.3rem 0 0.6rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Summary box */
.summary-box {
    background: #f5f9fd;
    border: 1px solid #d0e4f0;
    border-left: 4px solid #7b88d4;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.87rem; line-height: 1.65;
    color: #334155; margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Download button */
.dl-btn-wrap { margin-top: 1rem; text-align: center; }
a.dl-btn {
    display: inline-block;
    background: linear-gradient(135deg, #5BAD7E, #4a9469);
    color: white !important; text-decoration: none;
    padding: 0.75rem 2.2rem; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem; letter-spacing: 0.03em;
    transition: opacity 0.2s; cursor: pointer;
    box-shadow: 0 2px 10px rgba(91,173,126,0.25);
}
a.dl-btn:hover { opacity: 0.88; }

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #e2e8f0 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# Note: Ensure these files exist in your /data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sentiment_results.csv"
SUMMARY_PATH = BASE_DIR / "data" / "summaries.json"

try:
    results = pd.read_csv(DATA_PATH)
    results['predicted_sentiment'] = results['predicted_sentiment'].str.lower()

    with open(SUMMARY_PATH) as f:
        summaries = json.load(f)
    emotion_summaries = summaries["emotion_summaries"]

    counts = results['predicted_sentiment'].value_counts()
    total = len(results)
    pos = counts.get('positive', 0)
    neg = counts.get('negative', 0)
    neu = counts.get('neutral', 0)
    sentiments = list(counts.index)

    # ─────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────
    st.markdown('<div class="dash-title">📊 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # KPI CARDS
    # ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card total">
        <div class="kpi-label">Total Comments</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-pct">All analysed records</div>
        <div class="kpi-icon">💬</div>
      </div>
      <div class="kpi-card pos">
        <div class="kpi-label">Positive Sentiment</div>
        <div class="kpi-value">{pos/total*100:.1f} %</div>
        <div class="kpi-pct">{pos:,} comments</div>
        <div class="kpi-icon">😊</div>
      </div>
      <div class="kpi-card neu">
        <div class="kpi-label">Neutral Sentiment</div>
        <div class="kpi-value">{neu/total*100:.1f} %</div>
        <div class="kpi-pct">{neu:,} comments</div>
        <div class="kpi-icon">😐</div>
      </div>
      <div class="kpi-card neg">
        <div class="kpi-label">Negative Sentiment</div>
        <div class="kpi-value">{neg/total*100:.1f} %</div>
        <div class="kpi-pct">{neg:,} comments</div>
        <div class="kpi-icon">😠</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # CHARTS — Plotly Bar + Smaller Pie
    # ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        bar_colors_hex = [PALETTE.get(s, '#6366f1') for s in sentiments]
        pct_vals = [round(v / total * 100, 1) for v in counts.values]

        fig_bar = go.Figure(go.Bar(
            x=sentiments,
            y=list(counts.values),
            marker_color=bar_colors_hex,
            marker_line_width=0,
            text=[f'{p}%' for p in pct_vals],
            textposition='outside',
            textfont=dict(size=12, color='#334155', family='Sora, sans-serif'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata}%<extra></extra>',
            customdata=pct_vals,
        ))
        fig_bar.update_layout(
            title=dict(text='Sentiment Breakdown', font=dict(size=13, color='#334155', family='Sora, sans-serif'), x=0.5),
            paper_bgcolor='#f5f9fd',
            plot_bgcolor='#EAF3FB',
            font=dict(family='Sora, sans-serif', color='#334155'),
            xaxis=dict(showgrid=False, linecolor='#e2e8f0', tickfont=dict(size=12)),
            yaxis=dict(gridcolor='#e2e8f0', linecolor='#e2e8f0', tickfont=dict(size=10)),
            margin=dict(l=40, r=20, t=50, b=40),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
        fig2.patch.set_facecolor('#f5f9fd')
        ax2.set_facecolor('#f5f9fd')

        pie_colors = [PALETTE.get(s, '#6366f1') for s in sentiments]
        wedges, texts, autotexts = ax2.pie(
            counts.values,
            labels=None,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=2, edgecolor='white')
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color('white')
            at.set_fontweight('bold')

        legend_patches = [mpatches.Patch(color=PALETTE.get(s, '#6366f1'), label=s.capitalize()) for s in sentiments]
        ax2.legend(handles=legend_patches, loc='lower center', ncol=3,
                   framealpha=0, labelcolor='#334155', fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.1))
        ax2.set_title('Sentiment Distribution', color='#334155', fontsize=10, pad=6)
        fig2.tight_layout(pad=0.5)
        st.pyplot(fig2)
        plt.close(fig2)

    # ─────────────────────────────────────────────
    # OVERALL SUMMARY
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">🧾 Overall Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{summaries["overall_summary"]}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SENTIMENT-WISE SUMMARY
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">📝 Sentiment-wise Summary</div>', unsafe_allow_html=True)

    selected_sentiment = st.selectbox(
        "Select sentiment",
        options=list(emotion_summaries.keys()),
        label_visibility="collapsed",
        key="summary_select"
    )

    s_color   = PALETTE.get(selected_sentiment, '#6366f1')
    s_bg      = BG_PALE.get(selected_sentiment, '#f8fafc')
    s_txt_map = {'positive': '#155724', 'negative': '#7f1d1d', 'neutral': '#7a4f00'}
    s_txt     = s_txt_map.get(selected_sentiment, '#334155')

    st.markdown(
        f'<div class="summary-box" style="'
        f'border-left-color:{s_color};'
        f'background:{s_bg};'
        f'color:{s_txt};'
        f'">{emotion_summaries[selected_sentiment]}</div>',
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────
    # WORD CLOUD
    # ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Word Cloud</div>', unsafe_allow_html=True)

    selected_wc = st.selectbox(
        "Select sentiment for WordCloud",
        options=['all'] + [s for s in ['positive', 'neutral', 'negative'] if s in results['predicted_sentiment'].unique()],
        format_func=lambda x: x.capitalize(),
        label_visibility="collapsed",
        key="wc_select"
    )

    # Show colored badge
    wc_color_hex = PALETTE.get(selected_wc, '#7b88d4')
    wc_bg_hex    = BG_PALE.get(selected_wc, '#f5f9fd')
    badge_txt    = {'positive': '😊 Positive', 'negative': '😠 Negative', 'neutral': '😐 Neutral', 'all': '🌈 All'}
    badge_color  = '#7b88d4' if selected_wc == 'all' else wc_color_hex
    badge_bg     = '#eef0fc' if selected_wc == 'all' else wc_bg_hex
    
    st.markdown(
        f'<div style="display:inline-block;padding:0.25rem 0.9rem;border-radius:999px;'
        f'background:{badge_bg};border:1.5px solid {badge_color};'
        f'color:{badge_color};font-size:0.8rem;font-weight:600;margin-bottom:0.5rem;">'
        f'{badge_txt.get(selected_wc, selected_wc.capitalize())}</div>',
        unsafe_allow_html=True
    )

    if selected_wc == 'all':
        text_data = " ".join(results["comment_text"].dropna().astype(str))
        wc_bg_hex = '#f5f9fd'
    else:
        text_data = " ".join(results[results['predicted_sentiment'] == selected_wc]["comment_text"].dropna().astype(str))

    if text_data.strip():
        if selected_wc == 'all':
            sentiment_word_sets = {}
            for s in ['positive', 'neutral', 'negative']:
                s_text = " ".join(results[results['predicted_sentiment'] == s]["comment_text"].dropna().astype(str))
                sentiment_word_sets[s] = set(s_text.lower().split())

            palette_all = {'positive': (91, 173, 126), 'neutral': (196, 154, 60), 'negative': (192, 87, 78)}

            def color_func_all(word, *args, **kwargs):
                import random
                w = word.lower()
                for s, words in sentiment_word_sets.items():
                    if w in words:
                        r2, g2, b2 = palette_all[s]
                        shade = random.uniform(0.7, 1.0)
                        return f"rgb({int(r2*shade)},{int(g2*shade)},{int(b2*shade)})"
                return "rgb(120,136,212)"

            wc = WordCloud(width=900, height=340, background_color=wc_bg_hex, stopwords=STOPWORDS, max_words=120, color_func=color_func_all).generate(text_data)
        else:
            wc_hex = wc_color_hex.lstrip('#')
            r, g, b = tuple(int(wc_hex[i:i+2], 16) for i in (0, 2, 4))
            def color_func(*args, **kwargs):
                import random
                shade = random.uniform(0.55, 1.0)
                return f"rgb({int(r*shade)},{int(g*shade)},{int(b*shade)})"
            wc = WordCloud(width=900, height=340, background_color=wc_bg_hex, stopwords=STOPWORDS, max_words=120, color_func=color_func).generate(text_data)

        fig_wc, ax_wc = plt.subplots(figsize=(9, 3.4))
        fig_wc.patch.set_facecolor(wc_bg_hex)
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    else:
        st.warning("No comments available for this sentiment.")

    # ─────────────────────────────────────────────
    # DOWNLOAD SECTION
    # ─────────────────────────────────────────────
    csv_bytes = results.to_csv(index=False).encode('utf-8')
    b64_csv = base64.b64encode(csv_bytes).decode()
    st.markdown(
        f'<div class="dl-btn-wrap"><a class="dl-btn" href="data:text/csv;base64,{b64_csv}" '
        f'download="sentiment_analysis_results.csv">⬇ Download Complete Analysis Results</a></div>',
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("Data files not found. Please ensure 'sentiment_results.csv' and 'summaries.json' are in the data folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")