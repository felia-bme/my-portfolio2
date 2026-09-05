import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------
# 1. FILE PATHS & CONSTANTS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"
DATA_FILE = BASE_DIR / "breast_cancer_data.csv"

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

TARGET_CANDIDATES = ["diagnosis", "target", "label", "class", "diagnosis_target", "y"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----------------------------------------------------
# 2. PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Data & AI Portfolio | Breast Cancer Classification",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    .stButton > button {
        background-color: #1f5f9f;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #174a7c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 3. CACHED DATA & MODEL LOADERS
# ----------------------------------------------------
@st.cache_resource
def load_model():
    """Load model.pkl. Kalau belum ada (misal deploy pertama kali / storage
    ephemeral), latih otomatis sekali dari breast_cancer_data.csv supaya app
    tidak perlu langkah manual terpisah."""
    if not MODEL_FILE.exists():
        if not DATA_FILE.exists():
            return None
        with st.spinner("model.pkl belum ada — melatih model sekali di awal (±10-30 detik)..."):
            import train_model
            train_model.main()
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_training_data():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)


# ----------------------------------------------------
# 4. HELPER & PREPROCESSING FUNCTIONS
# ----------------------------------------------------
def find_target_column(df):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def prepare_uploaded_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    normalized = {c.lower(): c for c in df.columns}

    if all(col.lower() in normalized for col in FEATURE_NAMES):
        X = df[[normalized[col.lower()] for col in FEATURE_NAMES]].copy()
        target_col = find_target_column(df)
        y = df[target_col].copy() if target_col else None
        return X, y, target_col

    if len(df.columns) == 32:
        feature_block = df.iloc[:, 2:].copy()
        if feature_block.shape[1] == 30:
            X = feature_block.apply(pd.to_numeric, errors="coerce")
            y = df.iloc[:, 1].copy()
            return X, y, df.columns[1]

    if len(df.columns) == 30:
        X = df.apply(pd.to_numeric, errors="coerce")
        return X, None, None

    raise ValueError(
        "Format CSV tidak dikenali. Unggah file dengan 30 kolom fitur WDBC "
        "atau file WDBC lengkap 32 kolom (id, diagnosis, + 30 fitur)."
    )


def clean_target(y):
    """Ubah kolom target (B/M, Benign/Malignant, atau 0/1) menjadi 0/1."""
    if y is None:
        return None
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.unique().tolist())
        if unique.issubset({0, 1}):
            return numeric.astype(int)

    mapping = {"B": 0, "BENIGN": 0, "0": 0, "M": 1, "MALIGNANT": 1, "1": 1}
    cleaned = y.astype(str).str.strip().str.upper().map(mapping)
    if cleaned.isna().any():
        raise ValueError("Kolom target harus berisi nilai B/M, Benign/Malignant, atau 0/1.")
    return cleaned.astype(int)


def get_features_and_target(df):
    """Ambil X (30 fitur, urutan tetap) dan y (0/1) dari train_df, dipakai bersama
    oleh semua tab supaya evaluasi model selalu konsisten."""
    X = df[FEATURE_NAMES]
    target_col = "target" if "target" in df.columns else find_target_column(df)
    if target_col is None:
        raise ValueError("Tidak menemukan kolom target (diagnosis/target/label/class) di dataset.")
    y = clean_target(df[target_col])
    return X, y


def render_model_performance(model, X, y, key_prefix=""):
    """Menampilkan metrik, confusion matrix, ROC curve, learning curve, dan
    feature importance. Dipanggil dari lebih dari satu tab agar kode tidak
    diduplikasi (dan tidak bisa saling tidak sinkron)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    c4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
    c5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

    st.caption(
        f"Dievaluasi pada {len(X_test)} sampel test (20% data, "
        f"tidak digunakan saat training — random_state={RANDOM_STATE})."
    )
    st.divider()

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Benign", "Malignant"]
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        customdata = np.dstack([cm, cm_pct])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            customdata=customdata,
            colorscale="Blues",
            showscale=True,
            hovertemplate=(
                "True: %{y}<br>Predicted: %{x}"
                "<br>Count: %{customdata[0]}"
                "<br>% of true class: %{customdata[1]:.1f}%<extra></extra>"
            ),
            texttemplate="%{z}",
            textfont={"size": 18},
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis_autorange="reversed",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc_val = roc_auc_score(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc_val:.3f})",
            line=dict(color="#1f5f9f", width=3),
            customdata=thresholds,
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{customdata:.3f}<extra></extra>",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
            line=dict(color="gray", width=1, dash="dash"),
            hoverinfo="skip",
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis_range=[0, 1], yaxis_range=[0, 1.05],
            height=380,
            legend=dict(x=0.55, y=0.05),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=["Benign", "Malignant"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.divider()

    # Feature importance (kalau model mendukungnya)
    importances = None
    final_estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(final_estimator, "feature_importances_"):
        importances = pd.Series(final_estimator.feature_importances_, index=FEATURE_NAMES)
    elif hasattr(final_estimator, "coef_"):
        importances = pd.Series(np.abs(final_estimator.coef_[0]), index=FEATURE_NAMES)

    if importances is not None:
        st.subheader("Fitur Paling Berpengaruh")
        n_top = st.slider("Jumlah fitur ditampilkan", min_value=5, max_value=30, value=10, key="n_top_features")
        top_n = importances.sort_values(ascending=False).head(n_top).sort_values()
        fig_imp = go.Figure(go.Bar(
            x=top_n.values, y=top_n.index, orientation="h",
            marker_color="#1f5f9f",
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            xaxis_title="Importance",
            height=max(300, 28 * n_top),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.divider()

    st.subheader("Learning Curve")
    with st.spinner("Menghitung learning curve (cross-validation)..."):
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=model, X=X, y=y, cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
        )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig_lc = go.Figure()
    # Ribbon +/- 1 std, di-render sebagai area terisi transparan
    fig_lc.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill="toself", fillcolor="rgba(31,95,159,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig_lc.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill="toself", fillcolor="rgba(217,83,79,0.15)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
    ))
    fig_lc.add_trace(go.Scatter(
        x=train_sizes, y=train_mean, mode="lines+markers", name="Training Accuracy",
        line=dict(color="#1f5f9f", width=3), marker=dict(size=7),
        hovertemplate="Samples: %{x:.0f}<br>Train acc: %{y:.4f}<extra></extra>",
    ))
    fig_lc.add_trace(go.Scatter(
        x=train_sizes, y=val_mean, mode="lines+markers", name="Validation Accuracy",
        line=dict(color="#d9534f", width=3), marker=dict(size=7),
        hovertemplate="Samples: %{x:.0f}<br>Val acc: %{y:.4f}<extra></extra>",
    ))
    fig_lc.update_layout(
        xaxis_title="Training Samples",
        yaxis_title="Accuracy",
        height=420,
        hovermode="x unified",
        legend=dict(x=0.6, y=0.05),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_lc, use_container_width=True)


# ----------------------------------------------------
# 5. NAVIGATION TABS
# ----------------------------------------------------
tab_portfolio, tab_predictor, tab_analytics = st.tabs([
    "🏠 Portfolio & Profile",
    "🤖 Breast Cancer Predictor",
    "📊 EDA & Model Performance",
])

model = load_model()
train_df = load_training_data()


# ====================================================
# TAB 1: PORTFOLIO & FEATURED PROJECTS
# ====================================================
with tab_portfolio:
    st.title("My Data Science & AI Portfolio 🚀")
    st.write(
        "Welcome to my interactive portfolio! This web app showcases my background, "
        "key projects, and a live machine learning pipeline for medical diagnostic classification."
    )
    st.divider()

    st.subheader("About Me")
    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        if os.path.exists("images/profile.jpg"):
            st.image("images/profile.jpg", width=200)
        else:
            st.image("https://via.placeholder.com/200x200.png?text=Profile+Photo", width=200)

    with col_bio:
        st.markdown("""
        ### **Your Name**
        **Data Scientist & AI Specialist**

        Results-driven Data Professional skilled in end-to-end Machine Learning pipelines,
        exploratory data analysis, and deploying interactive diagnostic web platforms. Experienced in turning
        complex datasets into accurate, actionable predictions.
        """)
        st.text("Core Skills: Python · Scikit-Learn · Streamlit · Medical Informatics · Data Visualization")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Projects Completed", "5+")
    m2.metric("Features Handled", "30+")
    m3.metric("Model ROC-AUC Score", "0.99+")
    m4.metric("Dataset Samples", str(len(train_df)) if train_df is not None else "569")

    st.divider()

    st.subheader("Featured Projects")
    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown("### 🩺 Breast Cancer Wisconsin Classification")
            st.write("""
            Supervised ML model predicting tumor malignancy using 30 cell nucleus characteristics
            from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
            """)
            if os.path.exists("images/project1.png"):
                st.image("images/project1.png", use_container_width=True)
            else:
                st.caption("*(WDBC Diagnostic Classification)*")

            st.write("**Classification Model · Accuracy ≈ 97% · ROC-AUC ≈ 0.99**")
            st.info("👉 Try it live in the **🤖 Breast Cancer Predictor** tab.")

    with p2:
        with st.container(border=True):
            st.markdown("### 🏡 Real Estate Price Estimator")
            st.write("""
            Regression model predicting property market value based on land size, building area,
            location indices, and certificate status.
            """)
            if os.path.exists("images/project2.png"):
                st.image("images/project2.png", use_container_width=True)
            else:
                st.caption("*(Jakarta Property Market Model)*")

            st.write("**Random Forest Regressor · R² = 91%**")
            if st.button("Project 2 Details", use_container_width=True):
                st.info("Trained on 10,000+ real estate property listings.")

    with p3:
        with st.container(border=True):
            st.markdown("### 💬 Sentiment Analysis Pipeline")
            st.write("""
            Automated Natural Language Processing (NLP) pipeline classifying customer product reviews
            into sentiment categories.
            """)
            if os.path.exists("images/project3.png"):
                st.image("images/project3.png", use_container_width=True)
            else:
                st.caption("*(E-Commerce Sentiment Engine)*")

            st.write("**TF-IDF + Logistic Regression · F1 = 87%**")
            if st.button("Project 3 Details", use_container_width=True):
                st.info("Processed over 50,000 user reviews.")


# ====================================================
# TAB 2: PREDICTOR (single-sample form + batch CSV upload)
# ====================================================
with tab_predictor:
    if model is None or train_df is None:
        st.warning("Please make sure `model.pkl` and `breast_cancer_data.csv` are present in the directory. "
                    "Run `python train_model.py` first to generate `model.pkl`.")
    else:
        try:
            X_all, y_all = get_features_and_target(train_df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # ------------------------------------------------
        # A. Single-sample manual predictor
        # ------------------------------------------------
        st.header("🔍 Prediksi Satu Sampel (Input Manual)")
        st.write("Geser slider di bawah untuk mengubah nilai fitur, lalu lihat hasil prediksi model secara langsung.")

        feature_stats = X_all.describe()
        with st.expander("Atur nilai fitur", expanded=True):
            cols = st.columns(3)
            user_values = {}
            for i, feat in enumerate(FEATURE_NAMES):
                col = cols[i % 3]
                lo, hi = float(feature_stats.loc["min", feat]), float(feature_stats.loc["max", feat])
                default = float(feature_stats.loc["50%", feat])
                user_values[feat] = col.slider(
                    feat, min_value=lo, max_value=hi, value=default,
                    key=f"slider_{feat}",
                )

        if st.button("🩺 Prediksi Sampel Ini", use_container_width=True):
            sample_df = pd.DataFrame([user_values])[FEATURE_NAMES]
            pred = model.predict(sample_df)[0]
            prob = model.predict_proba(sample_df)[0, 1]

            if pred == 1:
                st.error(f"**Hasil: Malignant (Ganas)** — probabilitas malignant: {prob:.1%}")
            else:
                st.success(f"**Hasil: Benign (Jinak)** — probabilitas malignant: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))

        st.divider()

        # ------------------------------------------------
        # B. Batch CSV predictor
        # ------------------------------------------------
        st.header("📁 Prediksi Batch dari File CSV")
        st.write("Unggah CSV berisi 30 kolom fitur WDBC (atau format lengkap 32 kolom) untuk prediksi banyak sampel sekaligus.")

        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"], key="batch_uploader")

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)

                with st.expander("📄 Pratinjau Data Mentah", expanded=False):
                    st.dataframe(raw_df.head(), use_container_width=True)

                X_user, y_user, target_col = prepare_uploaded_data(raw_df)

                if X_user.shape[1] != 30:
                    raise ValueError(f"Model membutuhkan 30 fitur, tapi {X_user.shape[1]} yang tersedia.")

                X_user.columns = FEATURE_NAMES
                X_user = X_user.apply(pd.to_numeric, errors="coerce")

                if X_user.isna().any().any():
                    bad_cols = X_user.columns[X_user.isna().any()].tolist()
                    raise ValueError(f"Ditemukan nilai kosong/non-numerik pada kolom: {', '.join(bad_cols)}")

                predictions = model.predict(X_user)
                probabilities = model.predict_proba(X_user)[:, 1]

                result_df = raw_df.copy()
                result_df["Prediction"] = np.where(predictions == 1, "Malignant", "Benign")
                result_df["Malignant_Probability"] = probabilities.round(4)

                st.subheader("Hasil Prediksi")
                n_malignant = int((predictions == 1).sum())
                st.caption(f"{len(result_df)} sampel diproses — {n_malignant} diprediksi Malignant, "
                           f"{len(result_df) - n_malignant} diprediksi Benign.")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    column_config={
                        "Malignant_Probability": st.column_config.ProgressColumn(
                            "Malignant Probability", min_value=0.0, max_value=1.0, format="%.2f",
                        )
                    },
                )

                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="breast_cancer_predictions.csv",
                    mime="text/csv",
                )

                # Kalau file yang diunggah punya label asli, tampilkan evaluasi cepat
                if y_user is not None:
                    try:
                        y_true = clean_target(y_user)
                        acc = accuracy_score(y_true, predictions)
                        st.info(f"File ini menyertakan label asli (`{target_col}`) — akurasi pada data ini: {acc:.3f}")
                    except ValueError:
                        pass

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")


# ====================================================
# TAB 3: DATA EXPLORATION & MODEL PERFORMANCE
# ====================================================
with tab_analytics:
    st.title("📊 Dataset Exploratory Analysis & Model Evaluation")
    st.divider()

    if train_df is None:
        st.warning("⚠️ `breast_cancer_data.csv` not found in root directory. Showing interactive placeholder analytics.")

        np.random.seed(42)
        sample_df = pd.DataFrame({
            "radius_mean": np.random.normal(14, 3, 200),
            "texture_mean": np.random.normal(19, 4, 200),
            "perimeter_mean": np.random.normal(91, 20, 200),
            "diagnosis": np.random.choice(["B", "M"], 200),
        })
        fig = px.histogram(sample_df, x="radius_mean", color="diagnosis", marginal="box", title="Feature Distribution (Demo)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        tab_eda, tab_perf = st.tabs(["📈 Visualisasi Data (EDA)", "🧪 Performa Model Machine Learning"])

        # Sub-tab 1: EDA
        with tab_eda:
            try:
                X_eda, y_eda = get_features_and_target(train_df)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            st.subheader("Dataset Overview")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", len(train_df))
            c2.metric("Total Features", len(FEATURE_NAMES))
            c3.metric("Malignant Cases", int((y_eda == 1).sum()))

            st.divider()
            col_dist, col_corr = st.columns([1, 1])

            with col_dist:
                st.subheader("Feature Distribution")
                selected_feature = st.selectbox("Select Feature:", FEATURE_NAMES, key="distribution_feature")
                plot_df = X_eda.copy()
                plot_df["Diagnosis"] = np.where(y_eda == 1, "Malignant", "Benign")
                fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
                sns.histplot(
                    data=plot_df, x=selected_feature, hue="Diagnosis", kde=True,
                    ax=ax_dist, bins=30, palette={"Benign": "#1f5f9f", "Malignant": "#d9534f"},
                )
                ax_dist.set_title(f"Distribution of {selected_feature}")
                st.pyplot(fig_dist, clear_figure=True)

            with col_corr:
                st.subheader("Correlation Matrix")
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                corr = X_eda[FEATURE_NAMES[:10]].corr()
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax_corr)
                ax_corr.set_title("Correlation Heatmap (Mean Features)")
                st.pyplot(fig_corr, clear_figure=True)

        # Sub-tab 2: Performance Evaluation
        with tab_perf:
            if model is None:
                st.warning("Please make sure `model.pkl` is loaded to view training metrics.")
            else:
                st.header("PERFORMA MODEL")
                try:
                    X_perf, y_perf = get_features_and_target(train_df)
                    render_model_performance(model, X_perf, y_perf, key_prefix="perf")
                except ValueError as e:
                    st.error(str(e))