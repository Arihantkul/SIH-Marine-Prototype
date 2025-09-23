import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="SAGAR DARPAN",
    page_icon="🌊",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* Sidebar background and text */
.stSidebar {
    background-color: #003366 !important;
    color: white !important;
}

/* Sidebar labels, radio buttons, headers */
.stSidebar .stRadio > div > label,
.stSidebar .stMarkdown,
.stSidebar h2,
.stSidebar h3,
.stSidebar p {
    color: white !important;
}

/* Fix for radio button text */
.stRadio > div > label {
    color: white !important;
}

/* Main panel */
.main { background-color: #e0f0ff; }

/* Header titles */
h1, h2, h3 { color: #004e89; font-family: 'Segoe UI', sans-serif; }

/* Big title */
.big-title { font-size: 40px; font-weight: 700; color: #004e89; }

/* Tagline */
.tagline { font-size: 18px; color: #0077b6; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div style="display: flex; align-items: center; gap: 10px;">
    <div class="logo">🌊</div>
    <div>
        <div class="big-title">SAGAR DARPAN</div>
        <div class="tagline">AI-Driven Marine Data Platform</div>
    </div>
</div>
<hr style="margin:10px 0; border-top: 2px solid #0077b6;">
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("Select Module")
module = st.sidebar.radio("Modules", [
    "Fish Count",
    "Fish Size Classification",
    "Biodiversity Clustering",
    "eDNA Analysis",
    "Threat Meter",
    "Fisheries Sustainability Index (FSI)",
    "Heatmap (Temp vs Salinity)"
])

# ------------------ DATA ------------------
BASE_DIR = r"C:\Users\aa\Desktop\Coding\sih-marine-prototype"
DATA_PATH = os.path.join(BASE_DIR, "data", "fish_data.csv")

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df = load_data(DATA_PATH)
if df is None:
    st.warning("Dataset not found. Using generated sample data.")
    df = pd.DataFrame({
        "Temperature (°C)": np.random.uniform(24,30,200),
        "Salinity (PSU)": np.random.uniform(33,37,200),
        "Count": np.random.randint(1,20,200),
        "Fish Length (cm)": np.random.uniform(5,100,200),
        "Weight (g)": np.random.uniform(50,5000,200),
        "Species": np.random.choice(["Species A","Species B","Species C"],200)
    })

# ------------------ MODEL PATHS ------------------
models_dir = os.path.join(BASE_DIR, "models")
per_species_dir = os.path.join(models_dir, "per_species")
per_length_dir = os.path.join(models_dir, "per_length")
per_weight_dir = os.path.join(models_dir, "per_weight")

# ------------------ MODULES ------------------
if module == "Fish Count":
    st.header("🐟 Fish Count Prediction")
    temp = st.slider("Temperature (°C)", 24.0, 30.0, 27.0, step=0.1)
    sal = st.slider("Salinity (PSU)", 33.0, 37.0, 35.0, step=0.1)

    # Model Type selection (radio buttons)
    st.subheader("Select Prediction Model")
    model_type = st.radio("Model Type", ["General", "By Species", "By Length", "By Weight"])

    if model_type == "By Species":
        species_models = [f for f in os.listdir(per_species_dir) if f.endswith(".pkl")]
        species_names = [f.replace("fish_count_model_", "").replace(".pkl", "") for f in species_models]
        species = st.radio("Select Species", species_names)
        model_path = os.path.join(per_species_dir, f"fish_count_model_{species}.pkl")
        st.info(f"Using model: {os.path.basename(model_path)}")
    elif model_type == "By Length":
        length_labels = ["short", "medium", "long"]
        length = st.radio("Select Length Category", length_labels)
        model_path = os.path.join(per_length_dir, f"fish_count_model_length_{length}.pkl")
        st.info(f"Using model: {os.path.basename(model_path)}")
    elif model_type == "By Weight":
        weight_labels = ["light", "medium", "heavy"]
        weight = st.radio("Select Weight Category", weight_labels)
        model_path = os.path.join(per_weight_dir, f"fish_count_model_weight_{weight}.pkl")
        st.info(f"Using model: {os.path.basename(model_path)}")
    else:
        model_path = os.path.join(models_dir, "fish_count_model.pkl")
        st.info(f"Using model: {os.path.basename(model_path)}")

    if st.button("Predict Fish Count"):
        model = joblib.load(model_path)
        X = pd.DataFrame([[temp, sal]], columns=["Temperature (°C)", "Salinity (PSU)"])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Fish Count: {prediction:.2f}")

        # Comparison Graphs
        compare_data = []
        if model_type == "By Species":
            for sp, fname in zip(species_names, species_models):
                m = joblib.load(os.path.join(per_species_dir, fname))
                pred = m.predict(X)[0]
                compare_data.append({"Category": sp, "Predicted Count": pred})
        elif model_type == "By Length":
            for l in length_labels:
                mpath = os.path.join(per_length_dir, f"fish_count_model_length_{l}.pkl")
                if os.path.exists(mpath):
                    m = joblib.load(mpath)
                    pred = m.predict(X)[0]
                    compare_data.append({"Category": l, "Predicted Count": pred})
        elif model_type == "By Weight":
            for w in weight_labels:
                mpath = os.path.join(per_weight_dir, f"fish_count_model_weight_{w}.pkl")
                if os.path.exists(mpath):
                    m = joblib.load(mpath)
                    pred = m.predict(X)[0]
                    compare_data.append({"Category": w, "Predicted Count": pred})
        else:
            compare_data.append({"Category": "General", "Predicted Count": prediction})
        
        df_compare = pd.DataFrame(compare_data)
        st.bar_chart(df_compare.set_index("Category"))

# ---------- Other modules remain unchanged ----------
elif module == "Fish Size Classification":
    st.header("📏 Fish Size Classification")
    length = st.slider("Fish Length (cm)", 5.0, 100.0, 25.0, step=0.1)
    weight = st.slider("Fish Weight (g)", 50.0, 5000.0, 500.0, step=1.0)
    if length<15 and weight<200:
        size="Small"
    elif length<40 and weight<1000:
        size="Medium"
    else:
        size="Large"
    st.success(f"Predicted Size: {size}")

elif module == "Biodiversity Clustering":
    st.header("🧩 Biodiversity Clustering (KMeans)")
    data = df[["Fish Length (cm)", "Weight (g)"]].sample(min(100,len(df)))
    n_clusters = st.slider("Number of Clusters", 2, 6, 3, step=1)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    data["cluster"]=km.labels_
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=data, x="Fish Length (cm)", y="Weight (g)", hue="cluster", palette="tab10", s=100, ax=ax)
    ax.set_title("KMeans Clusters")
    st.pyplot(fig)

elif module == "eDNA Analysis":
    st.header("🧬 eDNA Analysis (Demo)")
    taxa = df["Species"].value_counts()
    st.bar_chart(taxa)

elif module == "Threat Meter":
    st.header("⚠️ Threat Meter")
    temp_anom = st.slider("Temperature anomaly (°C)", -2.0, 5.0, 0.5, step=0.1)
    fishing = st.slider("Fishing pressure", 0.0, 100.0, 40.0, step=1.0)
    pollution = st.slider("Pollution index", 0.0, 100.0, 20.0, step=1.0)
    score = max(0, min(100, temp_anom*12 + fishing*0.35 + pollution*0.2))
    status = "Low 🟢" if score<40 else ("Moderate 🟡" if score<70 else "High 🟠")
    st.metric("Threat Score", round(score,1), status)

elif module == "Fisheries Sustainability Index (FSI)":
    st.header("🌍 Fisheries Sustainability Index (FSI)")
    labels = ['Biomass','Biodiversity','CPUE','Habitat','Governance']
    values = [st.slider(l,0.0,100.0,50.0, step=1.0) for l in labels]
    fsi = np.dot(values,[0.25,0.2,0.2,0.15,0.2])
    status = "Sustainable ✅" if fsi>70 else ("Attention ⚠️" if fsi>50 else "Critical ❌")
    st.metric("FSI Score", round(fsi,1), status)

    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2, color='#0077b6')
    ax.fill(angles, values, alpha=0.25, color='#00b4d8')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    ax.set_title("FSI Radar")
    st.pyplot(fig)

elif module == "Heatmap (Temp vs Salinity)":
    st.header("🌡️ Heatmap: Temp vs Salinity")
    df_heat = df.copy()
    t_bins = np.round(np.linspace(df_heat["Temperature (°C)"].min(), df_heat["Temperature (°C)"].max(), 8),1)
    s_bins = np.round(np.linspace(df_heat["Salinity (PSU)"].min(), df_heat["Salinity (PSU)"].max(), 8),1)
    df_heat['t_bin'] = pd.cut(df_heat["Temperature (°C)"], bins=t_bins, include_lowest=True, labels=t_bins[:-1])
    df_heat['s_bin'] = pd.cut(df_heat["Salinity (PSU)"], bins=s_bins, include_lowest=True, labels=s_bins[:-1])
    pivot = df_heat.pivot_table(index='t_bin',columns='s_bin',values='Count',aggfunc='mean').fillna(0)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label':'Mean Count'}, ax=ax)
    ax.set_xlabel("Salinity bin")
    ax.set_ylabel("Temperature bin")
    ax.set_title("Mean Count Heatmap (Temp x Salinity)")
    st.pyplot(fig)
