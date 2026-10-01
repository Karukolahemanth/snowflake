
import streamlit as st
import pandas as pd
import numpy as np
import os, re, time, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* background */
.stApp { background: #0b0f1a; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0e1520 0%,#0b0f1a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 22px !important;
    transition: transform .2s, box-shadow .2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,.5);
}
[data-testid="metric-container"] label {
    color: #8899aa !important; font-size:.72rem !important;
    font-weight:600 !important; letter-spacing:.1em !important;
    text-transform:uppercase !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e8edf3 !important; font-size:1.9rem !important; font-weight:800 !important;
}

/* textarea */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8edf3 !important;
    font-size: .95rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.12) !important;
}
.stTextArea label { color: #8899aa !important; font-weight:600 !important; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight:700 !important;
    font-size: 1rem !important; padding: 14px 0 !important;
    width: 100% !important; transition: all .25s !important;
    box-shadow: 0 4px 20px rgba(37,99,235,.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#1d4ed8,#1e3a8a) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,.6) !important;
    transform: translateY(-2px) !important;
}

/* radio sidebar */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #cbd5e1 !important; font-size:.9rem !important;
    padding: 8px 12px !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,.06) !important;
}

/* selectbox */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #e8edf3 !important;
}

/* tabs */
button[data-baseweb="tab"] { color:#8899aa !important; font-weight:600 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color:#60a5fa !important;
    border-bottom: 3px solid #60a5fa !important;
}

/* dataframe */
[data-testid="stDataFrame"] iframe { border-radius: 12px !important; }

/* divider */
hr { border-color: rgba(255,255,255,.07) !important; }

/* spinner */
.stSpinner > div { border-top-color: #2563eb !important; }
</style>
""", unsafe_allow_html=True)

def card(content: str, border_color: str = "rgba(255,255,255,0.08)") -> str:
    return f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid {border_color};
                border-radius:16px; padding:28px 30px; margin:6px 0;">
        {content}
    </div>"""

def badge(text: str, color: str) -> str:
    bg = {"red":"rgba(239,68,68,.15)","green":"rgba(34,197,94,.15)","yellow":"rgba(234,179,8,.15)"}
    tc = {"red":"#f87171","green":"#4ade80","yellow":"#facc15"}
    bc = {"red":"rgba(239,68,68,.35)","green":"rgba(34,197,94,.35)","yellow":"rgba(234,179,8,.35)"}
    return (f'<span style="background:{bg[color]};color:{tc[color]};border:1px solid {bc[color]};'
            f'border-radius:30px;padding:4px 14px;font-size:.8rem;font-weight:700;">{text}</span>')

DATA_DIR = r"C:\Users\DELL 7410\Downloads\kaggle_spam_data"

@st.cache_data(show_spinner=False)
def load_dataset(max_rows: int = 30_000):
    """Load & merge Enron + CEAS_08 + SpamAssasin datasets."""
    frames = []

    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "Enron.csv"),
                         encoding="utf-8", on_bad_lines="skip",
                         usecols=["subject","body","label"])
        df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
        frames.append(df[["text","label"]])
    except Exception as e:
        st.warning(f"Enron.csv skipped: {e}")

    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "CEAS_08.csv"),
                         encoding="utf-8", on_bad_lines="skip",
                         usecols=["subject","body","label"])
        df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
        frames.append(df[["text","label"]])
    except Exception as e:
        st.warning(f"CEAS_08.csv skipped: {e}")

    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "SpamAssasin.csv"),
                         encoding="utf-8", on_bad_lines="skip",
                         usecols=["subject","body","label"])
        df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
        frames.append(df[["text","label"]])
    except Exception as e:
        st.warning(f"SpamAssasin.csv skipped: {e}")

    if not frames:
        st.error("❌ No dataset files found. Check DATA_DIR path.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text","label"])
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined = combined[combined["text"].str.len() > 5]
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(combined) > max_rows:
        ham_df  = combined[combined["label"]==0].head(max_rows//2)
        spam_df = combined[combined["label"]==1].head(max_rows//2)
        combined = pd.concat([ham_df, spam_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    return combined

@st.cache_resource(show_spinner=False)
def train_model():
    """Train TF-IDF + Random Forest pipeline on real email data."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score,
                                  confusion_matrix)

    df = load_dataset()
    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=25_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]\w+\b",
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "cm":        confusion_matrix(y_test, y_pred),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
    }

    return pipe, metrics, df

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 12px;">
        <div style="font-size:2.6rem;">🛡️</div>
        <h1 style="color:#e8edf3; font-size:1.2rem; font-weight:800; margin:6px 0 2px;">
            Email Spam Detector
        </h1>
        <p style="color:#8899aa; font-size:.75rem; margin:0;">
            Powered by Kaggle Datasets
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Check Email", "📊 Dataset Stats", "🤖 Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""
    <p style="color:#8899aa;font-size:.72rem;font-weight:600;letter-spacing:.08em;
               text-transform:uppercase;margin:0 0 8px;">📦 Data Sources</p>
    """, unsafe_allow_html=True)
    for ds, n in [("Enron Email", "~33k"), ("CEAS 2008", "~39k"), ("SpamAssasin", "~5.8k")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 0;border-bottom:1px solid rgba(255,255,255,.05);">
            <span style="color:#cbd5e1;font-size:.8rem;">{ds}</span>
            <span style="color:#60a5fa;font-size:.75rem;font-weight:700;">{n}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Model: TF-IDF + Logistic Regression")

@st.cache_data(show_spinner=False)
def get_model_once():
    return train_model()

if "model_loaded" not in st.session_state:
    with st.spinner("⚙️  Loading dataset & training model — this takes ~20 seconds on first run…"):
        pipe, metrics, df = get_model_once()
    st.session_state["model_loaded"] = True
else:
    pipe, metrics, df = get_model_once()

if page == "🔍 Check Email":

    st.markdown("""
    <div style="padding:0 0 8px;">
        <h1 style="color:#e8edf3;font-size:2rem;font-weight:900;margin:0;">
            🛡️ Is this email spam?
        </h1>
        <p style="color:#8899aa;font-size:1rem;margin:6px 0 0;">
            Paste the email subject and body below — our model will classify it instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    PRESETS = {
        "— Type your own —": ("", ""),
        "🚨 Classic Spam — Prize winner":
            ("Congratulations! You have WON $1,000,000!!!",
             "Dear Winner, You have been selected as the lucky winner of our international lottery. "
             "To claim your prize of ONE MILLION DOLLARS, please send your full name, bank account "
             "details and a small processing fee of $50 to our agent immediately. "
             "Click here now! http://claimprize-now.xyz/claim"),
        "🚨 Phishing — PayPal alert":
            ("URGENT: Your PayPal account has been suspended",
             "Your PayPal account has been flagged for suspicious activity and will be permanently "
             "closed in 24 hours unless you verify your identity. Click the link below immediately "
             "to restore access: http://paypal-verify-secure.net/login"),
        "🚨 Nigerian Fraud":
            ("CONFIDENTIAL BUSINESS PROPOSAL — $8.5M USD",
             "Dear Friend, I am Dr. James Okafor, a senior banking official in Lagos, Nigeria. "
             "I have a business proposal worth $8.5 million USD that requires your assistance. "
             "You will receive 30% for your cooperation. Please reply in utmost confidentiality."),
        "✅ Legitimate — Meeting invite":
            ("Team standup — Thursday 10am",
             "Hi all, just a reminder that our weekly standup is on Thursday at 10am in the "
             "main conference room. Agenda: sprint review, blocker discussion, and Q3 planning. "
             "Please come prepared with your updates. Thanks, Sarah"),
        "✅ Legitimate — Order shipped":
            ("Your Amazon order has been shipped",
             "Hello, your order #112-3456789 has been dispatched and is on its way. "
             "Estimated delivery is Friday between 2pm and 6pm. You can track your package "
             "using the tracking number TBA123456789. Thank you for shopping with us."),
    }

    preset_key = st.selectbox("⚡ Load a preset email", list(PRESETS.keys()), index=0)
    p_subj, p_body = PRESETS[preset_key]

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        subject_in = st.text_input(
            "📌 Subject line",
            value=p_subj,
            placeholder="Enter email subject…",
        )
        body_in = st.text_area(
            "📝 Email body",
            value=p_body,
            placeholder="Paste the full email body here…",
            height=280,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍  Analyse Email", use_container_width=True)

    with col_result:
        if analyse_btn:
            combined_text = f"{subject_in} {body_in}".strip()
            if len(combined_text) < 5:
                st.warning("⚠️  Please enter some email text first.")
            else:
                clean = preprocess(combined_text)
                proba = pipe.predict_proba([clean])[0][1]
                pred  = int(proba >= 0.5)

                if proba >= 0.85:
                    level, level_color, level_emoji = "Very High Confidence", "#ef4444", "🔴"
                elif proba >= 0.65:
                    level, level_color, level_emoji = "High Confidence",      "#f97316", "🟠"
                elif proba >= 0.50:
                    level, level_color, level_emoji = "Moderate",             "#eab308", "🟡"
                elif proba >= 0.35:
                    level, level_color, level_emoji = "Probably Safe",        "#84cc16", "🟢"
                else:
                    level, level_color, level_emoji = "Very Likely Safe",     "#22c55e", "✅"

                if pred == 1:
                    result_icon  = "🚨"
                    result_label = "SPAM / PHISHING"
                    result_desc  = "This email has strong spam or phishing indicators."
                    card_border  = "rgba(239,68,68,.4)"
                    card_bg      = "rgba(239,68,68,.07)"
                    bar_color    = "#ef4444"
                else:
                    result_icon  = "✅"
                    result_label = "LEGITIMATE"
                    result_desc  = "This email appears to be safe and genuine."
                    card_border  = "rgba(34,197,94,.4)"
                    card_bg      = "rgba(34,197,94,.07)"
                    bar_color    = "#22c55e"

                st.markdown(f"""
                <div style="background:{card_bg}; border:2px solid {card_border};
                            border-radius:20px; padding:32px 28px; text-align:center;
                            box-shadow:0 0 40px {card_border};">
                    <div style="font-size:3.5rem; margin-bottom:8px;">{result_icon}</div>
                    <h2 style="color:{bar_color}; font-size:1.8rem; font-weight:900;
                               letter-spacing:.04em; margin:0 0 6px;">{result_label}</h2>
                    <p style="color:#8899aa; font-size:.9rem; margin:0 0 24px;">{result_desc}</p>

                    <!-- Probability bar -->
                    <div style="margin:0 auto; max-width:320px;">
                        <div style="display:flex; justify-content:space-between;
                                    color:#8899aa; font-size:.72rem; margin-bottom:4px;">
                            <span>SAFE</span>
                            <span>SPAM</span>
                        </div>
                        <div style="background:rgba(255,255,255,.08); border-radius:30px;
                                    height:14px; overflow:hidden;">
                            <div style="width:{proba*100:.1f}%; height:100%;
                                        background:linear-gradient(90deg,#22c55e,{bar_color});
                                        border-radius:30px; transition:width .6s ease;">
                            </div>
                        </div>
                        <div style="text-align:center; margin-top:10px;">
                            <span style="color:{level_color}; font-size:2rem; font-weight:900;">
                                {proba*100:.1f}%
                            </span>
                            <span style="color:#8899aa; font-size:.8rem;"> spam probability</span>
                        </div>
                    </div>

                    <div style="margin-top:16px; color:{level_color};
                                font-size:.82rem; font-weight:700;">
                        {level_emoji} {level}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("""
                <p style="color:#8899aa;font-size:.72rem;font-weight:700;
                           letter-spacing:.1em;text-transform:uppercase;margin:0 0 10px;">
                    ⚡ Spam Signal Checklist
                </p>""", unsafe_allow_html=True)

                raw = combined_text.lower()
                signals = [
                    ("Urgent / threatening language",
                     any(w in raw for w in ["urgent","immediately","suspended","limited","act now","warning","attention"])),
                    ("Prize / money language",
                     any(w in raw for w in ["winner","won","prize","million","lottery","claim","free","reward","cash"])),
                    ("Suspicious URL present",
                     bool(re.search(r"http[s]?://", combined_text, re.I))),
                    ("Request for personal info",
                     any(w in raw for w in ["password","bank account","credit card","ssn","social security","verify","confirm"])),
                    ("Excessive punctuation / caps",
                     bool(re.search(r"[!]{2,}", combined_text) or sum(1 for c in combined_text if c.isupper()) > 30)),
                    ("Short / vague body",
                     len(body_in.split()) < 15),
                ]

                for sig_name, triggered in signals:
                    icon = "🔴" if triggered else "🟢"
                    color = "#f87171" if triggered else "#4ade80"
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:12px;
                                padding:8px 0; border-bottom:1px solid rgba(255,255,255,.05);">
                        <span style="font-size:1rem;">{icon}</span>
                        <span style="color:{'#f87171' if triggered else '#6b7280'};
                                     font-size:.85rem; font-weight:{'600' if triggered else '400'};">
                            {sig_name}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,.02); border:2px dashed rgba(255,255,255,.1);
                        border-radius:20px; padding:60px 30px; text-align:center; height:100%;">
                <div style="font-size:3rem; margin-bottom:16px;">📧</div>
                <h3 style="color:#4b5563; font-size:1.1rem; font-weight:600; margin:0 0 8px;">
                    Awaiting email input
                </h3>
                <p style="color:#374151; font-size:.88rem; margin:0;">
                    Type or paste an email on the left,<br>then click <strong>Analyse Email</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Dataset Stats":
    import plotly.express as px
    import plotly.graph_objects as go

    PLOT_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_x=0.5,
    )

    st.markdown("""
    <h1 style="color:#e8edf3;font-size:1.8rem;font-weight:900;margin:0 0 4px;">
        📊 Dataset Overview
    </h1>
    <p style="color:#8899aa;font-size:.95rem;margin:0 0 24px;">
        Merged from 3 real-world email datasets: Enron, CEAS 2008, SpamAssasin
    </p>
    """, unsafe_allow_html=True)

    total     = len(df)
    spam_n    = df["label"].sum()
    legit_n   = total - spam_n
    spam_pct  = spam_n / total * 100
    avg_len   = df["text"].str.len().mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📧 Total Emails",  f"{total:,}")
    c2.metric("🚨 Spam",          f"{spam_n:,}",  f"{spam_pct:.1f}%")
    c3.metric("✅ Legitimate",    f"{legit_n:,}", f"{100-spam_pct:.1f}%")
    c4.metric("📝 Avg Length",    f"{avg_len:,.0f} chars")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "📝 Text Analysis", "🔍 Sample Emails"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.pie(
                names=["Legitimate","Spam"],
                values=[legit_n, spam_n],
                color_discrete_sequence=["#22c55e","#ef4444"],
                hole=0.5,
                title="Spam vs Legitimate Distribution",
            )
            fig.update_traces(
                textfont_color="white",
                marker=dict(line=dict(color="#0b0f1a",width=3)),
            )
            fig.update_layout(**PLOT_THEME, title_font_size=15,
                              legend=dict(font=dict(color="#cbd5e1")))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            df["text_len"] = df["text"].str.len().clip(upper=5000)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=df[df["label"]==0]["text_len"], name="Legitimate",
                marker_color="#22c55e", opacity=0.7, nbinsx=40,
            ))
            fig2.add_trace(go.Histogram(
                x=df[df["label"]==1]["text_len"], name="Spam",
                marker_color="#ef4444", opacity=0.7, nbinsx=40,
            ))
            fig2.update_layout(
                **PLOT_THEME,
                title="Email Length Distribution",
                title_font_size=15,
                barmode="overlay",
                xaxis=dict(title="Character Count", gridcolor="rgba(255,255,255,.06)"),
                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,.06)"),
                legend=dict(font=dict(color="#cbd5e1")),
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        from collections import Counter
        import re as _re

        STOP = {"the","a","an","and","or","is","in","to","of","for","it","this",
                "that","be","are","was","have","with","you","your","i","on","at",
                "we","can","our","will","by","from","as","not","all","my","do",
                "url","re","subject","message","email","please","thank","regards"}

        def top_words(texts, n=20):
            tokens = []
            for t in texts:
                tokens.extend(w for w in _re.findall(r'\b[a-z]{3,}\b', t.lower())
                               if w not in STOP)
            return Counter(tokens).most_common(n)

        col_l, col_r = st.columns(2)
        with col_l:
            spam_words = top_words(df[df["label"]==1]["text"].head(3000))
            fig_sw = px.bar(
                x=[c for _,c in spam_words], y=[w for w,_ in spam_words],
                orientation="h",
                color=[c for _,c in spam_words],
                color_continuous_scale=["#991b1b","#ef4444"],
                title="Top 20 Words in Spam Emails",
                labels={"x":"Count","y":"Word"},
            )
            fig_sw.update_layout(**PLOT_THEME, title_font_size=14,
                                  showlegend=False,
                                  xaxis=dict(gridcolor="rgba(255,255,255,.06)"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,.06)"))
            st.plotly_chart(fig_sw, use_container_width=True)

        with col_r:
            legit_words = top_words(df[df["label"]==0]["text"].head(3000))
            fig_lw = px.bar(
                x=[c for _,c in legit_words], y=[w for w,_ in legit_words],
                orientation="h",
                color=[c for _,c in legit_words],
                color_continuous_scale=["#14532d","#22c55e"],
                title="Top 20 Words in Legitimate Emails",
                labels={"x":"Count","y":"Word"},
            )
            fig_lw.update_layout(**PLOT_THEME, title_font_size=14,
                                  showlegend=False,
                                  xaxis=dict(gridcolor="rgba(255,255,255,.06)"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,.06)"))
            st.plotly_chart(fig_lw, use_container_width=True)

    with tab3:
        st.markdown("#### 📋 Sample Emails from Dataset")
        filter_opt = st.selectbox("Show:", ["All","Spam Only","Legitimate Only"])
        sample_df = (df if filter_opt=="All"
                     else df[df["label"]==(1 if "Spam" in filter_opt else 0)])
        disp = sample_df[["text","label"]].head(50).copy()
        disp["label"] = disp["label"].map({0:"✅ Legitimate",1:"🚨 Spam"})
        disp["text"]  = disp["text"].str[:200] + "…"
        st.dataframe(disp.reset_index(drop=True),
                     column_config={"text": st.column_config.TextColumn("Email Preview", width="large"),
                                    "label": st.column_config.TextColumn("Label", width="small")},
                     hide_index=True, use_container_width=True)

elif page == "🤖 Model Performance":
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    PLOT_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        title_x=0.5,
    )

    st.markdown("""
    <h1 style="color:#e8edf3;font-size:1.8rem;font-weight:900;margin:0 0 4px;">
        🤖 Model Performance
    </h1>
    <p style="color:#8899aa;font-size:.95rem;margin:0 0 24px;">
        TF-IDF (25k features, bigrams) + Logistic Regression — trained on real Kaggle email data
    </p>
    """, unsafe_allow_html=True)

    m = metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🎯 Accuracy",  f"{m['accuracy']*100:.2f}%")
    c2.metric("📐 Precision", f"{m['precision']*100:.2f}%")
    c3.metric("🔁 Recall",    f"{m['recall']*100:.2f}%")
    c4.metric("🏆 F1 Score",  f"{m['f1']*100:.2f}%")
    c5.metric("📈 AUC-ROC",   f"{m['auc_roc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <p style="color:#8899aa;font-size:.72rem;font-weight:700;
                   letter-spacing:.1em;text-transform:uppercase;margin:0 0 10px;">
            Confusion Matrix
        </p>""", unsafe_allow_html=True)

        cm = m["cm"]
        labels = ["Legitimate","Spam"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale=[[0,"#0f172a"],[0.5,"#1e40af"],[1,"#ef4444"]],
            showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=26, color="white"),
        ))
        fig_cm.update_layout(
            **PLOT_THEME,
            height=340,
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        tn,fp,fn,tp = cm.ravel()
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">
            <div style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
                        border-radius:12px;padding:14px;text-align:center;">
                <p style="color:#8899aa;font-size:.68rem;margin:0;">TRUE POSITIVES</p>
                <p style="color:#4ade80;font-size:1.6rem;font-weight:800;margin:2px 0;">
                    {tp:,}</p>
                <p style="color:#6b7280;font-size:.7rem;margin:0;">Spam caught</p>
            </div>
            <div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);
                        border-radius:12px;padding:14px;text-align:center;">
                <p style="color:#8899aa;font-size:.68rem;margin:0;">FALSE NEGATIVES</p>
                <p style="color:#f87171;font-size:1.6rem;font-weight:800;margin:2px 0;">
                    {fn:,}</p>
                <p style="color:#6b7280;font-size:.7rem;margin:0;">Spam missed</p>
            </div>
            <div style="background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.25);
                        border-radius:12px;padding:14px;text-align:center;">
                <p style="color:#8899aa;font-size:.68rem;margin:0;">FALSE POSITIVES</p>
                <p style="color:#facc15;font-size:1.6rem;font-weight:800;margin:2px 0;">
                    {fp:,}</p>
                <p style="color:#6b7280;font-size:.7rem;margin:0;">Legit flagged</p>
            </div>
            <div style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
                        border-radius:12px;padding:14px;text-align:center;">
                <p style="color:#8899aa;font-size:.68rem;margin:0;">TRUE NEGATIVES</p>
                <p style="color:#4ade80;font-size:1.6rem;font-weight:800;margin:2px 0;">
                    {tn:,}</p>
                <p style="color:#6b7280;font-size:.7rem;margin:0;">Legit correct</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        metric_names  = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
        metric_values = [m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc_roc"]]
        colors        = ["#3b82f6","#8b5cf6","#ec4899","#f97316","#22c55e"]

        fig_bar = go.Figure()
        for name, val, color in zip(metric_names, metric_values, colors):
            fig_bar.add_trace(go.Bar(
                x=[name], y=[val],
                marker_color=color,
                marker_line_color="rgba(0,0,0,0)",
                text=f"{val*100:.2f}%",
                textposition="outside",
                textfont=dict(color="white", size=13),
                name=name,
            ))
        fig_bar.update_layout(
            **PLOT_THEME,
            title="All Metrics at a Glance",
            title_font_size=15,
            showlegend=False,
            yaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,.06)", title="Score"),
            xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
            height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
                    border-radius:16px;padding:20px 24px;margin-top:10px;">
            <p style="color:#8899aa;font-size:.72rem;font-weight:700;letter-spacing:.1em;
                       text-transform:uppercase;margin:0 0 12px;">⚙️ Model Configuration</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Algorithm</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        Logistic Regression</p>
                </div>
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Vectorizer</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        TF-IDF (1–2 grams)</p>
                </div>
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Max Features</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        25,000</p>
                </div>
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Training samples</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        {m['n_train']:,}</p>
                </div>
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Test samples</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        {m['n_test']:,}</p>
                </div>
                <div>
                    <p style="color:#6b7280;font-size:.72rem;margin:0;">Class balancing</p>
                    <p style="color:#e8edf3;font-size:.88rem;font-weight:600;margin:2px 0;">
                        Balanced weights</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<p style="text-align:center;color:#374151;font-size:.75rem;margin:0;">
    🛡️ Email Spam Detector &nbsp;·&nbsp;
    Kaggle Datasets: Enron, CEAS 2008, SpamAssasin &nbsp;·&nbsp;
    Model: TF-IDF + Logistic Regression &nbsp;·&nbsp; Snowflake ML Assignment
</p>
""", unsafe_allow_html=True)