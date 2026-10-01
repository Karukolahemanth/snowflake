import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="shield",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0b0f1a; }
[data-testid="stSidebar"] { background: #0e1520; border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 18px 22px !important;
}
[data-testid="metric-container"] label { color: #8899aa !important; font-size:.72rem !important; font-weight:600 !important; text-transform:uppercase !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e8edf3 !important; font-size:1.9rem !important; font-weight:800 !important; }
.stTextArea textarea { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; color: #e8edf3 !important; }
.stButton > button { background: linear-gradient(135deg,#2563eb,#1d4ed8) !important; color: white !important; border: none !important; border-radius: 12px !important; font-weight:700 !important; font-size: 1rem !important; padding: 14px 0 !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

session = get_active_session()

@st.cache_data(show_spinner=False)
def load_data():
    df = session.table("PHISHING_DB.PHISHING_SCHEMA.KAGGLE_EMAILS").to_pandas()
    df.columns = df.columns.str.lower()
    df["subject"] = df["subject"].fillna("")
    df["body"]    = df["body"].fillna("")
    df["text"]    = df["subject"] + " " + df["body"]
    df["label"]   = df["label"].astype(int)
    return df

@st.cache_resource(show_spinner=False)
def train_model():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    df = load_data()

    def clean(t):
        t = str(t).lower()
        t = re.sub(r"http\S+", " url ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() > 5]

    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2, token_pattern=r"\b[a-zA-Z]\w+\b")),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
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
    return pipe, metrics

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 12px;">
        <div style="font-size:2.4rem;">shield</div>
        <h1 style="color:#e8edf3;font-size:1.2rem;font-weight:800;margin:6px 0 2px;">Email Spam Detector</h1>
        <p style="color:#60a5fa;font-size:.75rem;margin:0;">Snowflake Native App</p>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", ["Check Email", "Dataset Stats", "Model Performance"], label_visibility="collapsed")
    st.divider()
    st.caption("Data: PHISHING_DB.PHISHING_SCHEMA.KAGGLE_EMAILS")
    st.caption("Model: TF-IDF + Logistic Regression")

with st.spinner("Loading Kaggle dataset from Snowflake & training model..."):
    pipe, metrics = train_model()
    df = load_data()

if page == "Check Email":
    st.markdown('<h1 style="color:#e8edf3;font-size:2rem;font-weight:900;margin:0;">Is this email spam?</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8899aa;font-size:1rem;">Paste email content below for instant classification</p>', unsafe_allow_html=True)
    st.divider()

    PRESETS = {
        "-- Type your own --": ("", ""),
        "Spam: Prize winner": ("Congratulations! You have WON $1,000,000!!!", "You have been selected as a lucky winner. Send your bank details immediately to claim your prize. Click: http://claimprize-now.xyz"),
        "Spam: PayPal phishing": ("URGENT: Your PayPal account suspended", "Your account has been flagged. Verify immediately at http://paypal-verify-secure.net/login or it will be deleted in 24 hours."),
        "Spam: Nigerian fraud": ("CONFIDENTIAL: $8.5M Business Proposal", "Dear Friend, I am Dr. James a senior banker. I have $8.5M USD requiring your assistance. You receive 30%. Reply confidentially."),
        "Legit: Meeting invite": ("Team standup Thursday 10am", "Hi all, weekly standup is Thursday at 10am in the main conference room. Agenda: sprint review and Q3 planning. Thanks, Sarah"),
        "Legit: Order shipped": ("Your Amazon order has been shipped", "Hello, your order #112-3456789 has been dispatched. Estimated delivery is Friday. Track with TBA123456789."),
    }

    preset = st.selectbox("Load a preset email", list(PRESETS.keys()))
    p_subj, p_body = PRESETS[preset]

    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        subj = st.text_input("Subject line", value=p_subj, placeholder="Enter email subject...")
        body = st.text_area("Email body", value=p_body, placeholder="Paste email body here...", height=280)
        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("Analyse Email", use_container_width=True)

    with col_r:
        if btn:
            text = f"{subj} {body}".strip()
            if len(text) < 5:
                st.warning("Please enter some email text first.")
            else:
                def clean(t):
                    t = str(t).lower()
                    t = re.sub(r"http\S+", " url ", t)
                    t = re.sub(r"[^a-z\s]", " ", t)
                    return re.sub(r"\s+", " ", t).strip()

                proba = pipe.predict_proba([clean(text)])[0][1]
                pred  = int(proba >= 0.5)

                if pred == 1:
                    label, icon, bcolor, tcolor = "SPAM / PHISHING", "SPAM", "rgba(239,68,68,.15)", "#ef4444"
                else:
                    label, icon, bcolor, tcolor = "LEGITIMATE", "SAFE", "rgba(34,197,94,.15)", "#22c55e"

                st.markdown(f"""
                <div style="background:{bcolor};border:2px solid {tcolor};border-radius:20px;padding:32px 28px;text-align:center;">
                    <div style="font-size:3rem;margin-bottom:8px;">{icon}</div>
                    <h2 style="color:{tcolor};font-size:1.8rem;font-weight:900;margin:0 0 6px;">{label}</h2>
                    <div style="margin:20px auto;max-width:300px;">
                        <div style="background:rgba(255,255,255,.08);border-radius:30px;height:14px;overflow:hidden;">
                            <div style="width:{proba*100:.1f}%;height:100%;background:linear-gradient(90deg,#22c55e,{tcolor});border-radius:30px;"></div>
                        </div>
                        <div style="margin-top:10px;">
                            <span style="color:{tcolor};font-size:2rem;font-weight:900;">{proba*100:.1f}%</span>
                            <span style="color:#8899aa;font-size:.8rem;"> spam probability</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br><p style='color:#8899aa;font-size:.72rem;font-weight:700;text-transform:uppercase;'>Spam Signal Checklist</p>", unsafe_allow_html=True)
                raw = text.lower()
                signals = [
                    ("Urgent/threatening language", any(w in raw for w in ["urgent","immediately","suspended","warning","act now"])),
                    ("Prize/money language",        any(w in raw for w in ["winner","won","prize","million","lottery","claim","free"])),
                    ("Suspicious URL",              bool(re.search(r"http[s]?://", text, re.I))),
                    ("Request for personal info",   any(w in raw for w in ["password","bank account","credit card","verify","confirm","ssn"])),
                    ("Excessive punctuation",       bool(re.search(r"[!]{2,}", text))),
                ]
                for name, hit in signals:
                    c, t = ("#f87171", "600") if hit else ("#6b7280", "400")
                    i = "RED" if hit else "GREEN"
                    st.markdown(f'<div style="padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);color:{c};font-size:.85rem;font-weight:{t};">[{i}] {name}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,.02);border:2px dashed rgba(255,255,255,.1);border-radius:20px;padding:60px 30px;text-align:center;height:100%;">
                <div style="font-size:3rem;margin-bottom:12px;">EMAIL</div>
                <p style="color:#4b5563;font-size:.9rem;">Type or paste an email on the left,<br>then click <strong>Analyse Email</strong></p>
            </div>""", unsafe_allow_html=True)

elif page == "Dataset Stats":
    import plotly.express as px
    import plotly.graph_objects as go
    THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", title_x=0.5)

    st.markdown('<h1 style="color:#e8edf3;font-size:1.8rem;font-weight:900;margin:0 0 20px;">Dataset Overview</h1>', unsafe_allow_html=True)
    total = len(df); spam = int(df["label"].sum()); legit = total - spam

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Emails",  f"{total:,}")
    c2.metric("Spam",          f"{spam:,}",  f"{spam/total*100:.1f}%")
    c3.metric("Legitimate",    f"{legit:,}", f"{legit/total*100:.1f}%")
    c4.metric("Sources",       "5 datasets")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.pie(names=["Legitimate","Spam"], values=[legit, spam],
                     color_discrete_sequence=["#22c55e","#ef4444"], hole=0.5, title="Distribution")
        fig.update_layout(**THEME, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        src = df.groupby("source")["label"].agg(["count","sum"]).reset_index()
        src.columns = ["Source","Total","Spam"]
        fig2 = px.bar(src, x="Source", y=["Spam","Total"], barmode="group",
                      color_discrete_sequence=["#ef4444","#3b82f6"], title="Emails by Source")
        fig2.update_layout(**THEME, height=360, xaxis=dict(gridcolor="rgba(255,255,255,.05)"), yaxis=dict(gridcolor="rgba(255,255,255,.05)"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Sample Emails from Snowflake")
    view_filter = st.selectbox("Filter", ["All","Spam","Legitimate"])
    sample = df if view_filter=="All" else df[df["label"]==(1 if view_filter=="Spam" else 0)]
    disp = sample[["subject","body","label","source"]].head(30).copy()
    disp["label"]  = disp["label"].map({0:"Legitimate",1:"Spam"})
    disp["body"]   = disp["body"].str[:150] + "..."
    st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)

elif page == "Model Performance":
    import plotly.graph_objects as go
    THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", title_x=0.5)

    st.markdown('<h1 style="color:#e8edf3;font-size:1.8rem;font-weight:900;margin:0 0 20px;">Model Performance</h1>', unsafe_allow_html=True)
    m = metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall",    f"{m['recall']*100:.2f}%")
    c4.metric("F1 Score",  f"{m['f1']*100:.2f}%")
    c5.metric("AUC-ROC",   f"{m['auc_roc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        cm = m["cm"]; tn,fp,fn,tp = cm.ravel()
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred Legit","Pred Spam"], y=["Act Legit","Act Spam"],
            colorscale=[[0,"#0f172a"],[1,"#ef4444"]], showscale=False,
            text=[[str(v) for v in r] for r in cm],
            texttemplate="<b>%{text}</b>", textfont=dict(size=28, color="white"),
        ))
        fig_cm.update_layout(**THEME, title="Confusion Matrix", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_r:
        names  = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
        values = [m["accuracy"],m["precision"],m["recall"],m["f1"],m["auc_roc"]]
        colors = ["#3b82f6","#8b5cf6","#ec4899","#f97316","#22c55e"]
        fig_b  = go.Figure([go.Bar(x=names, y=values, marker_color=colors,
                                    text=[f"{v*100:.1f}%" for v in values], textposition="outside",
                                    textfont=dict(color="white",size=13))])
        fig_b.update_layout(**THEME, title="All Metrics", height=320,
                             yaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,.06)"))
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown(f"""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:20px 24px;margin-top:16px;">
        <p style="color:#8899aa;font-size:.72rem;font-weight:700;text-transform:uppercase;margin:0 0 12px;">Model Info</p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Algorithm</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">Logistic Regression</p></div>
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Vectorizer</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">TF-IDF (1-2 grams)</p></div>
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Features</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">20,000</p></div>
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Training samples</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">{m['n_train']:,}</p></div>
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Test samples</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">{m['n_test']:,}</p></div>
            <div><p style="color:#6b7280;font-size:.7rem;margin:0;">Data source</p><p style="color:#e8edf3;font-weight:600;margin:2px 0;">Snowflake KAGGLE_EMAILS</p></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()
st.markdown('<p style="text-align:center;color:#374151;font-size:.75rem;">Email Spam Detector | Snowflake Native Streamlit | PHISHING_DB.PHISHING_SCHEMA</p>', unsafe_allow_html=True)
