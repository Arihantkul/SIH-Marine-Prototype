import streamlit as st
import pandas as pd
import joblib
import os

# --- Custom CSS for Aqua Theme ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a2d4f7 0%, #0e7fa6 100%);
        color: #00334e;
    }
    .stApp {
        background: linear-gradient(135deg, #a2d4f7 0%, #0e7fa6 100%);
    }
    .big-font {
        font-size:2.5rem !important;
        color: #0e7fa6;
        font-weight: bold;
        text-shadow: 1px 1px 2px #fff;
    }
    .fish-emoji {
        font-size: 2.5rem;
        margin-right: 0.5rem;
    }
    .aqua-box {
        background: rgba(255,255,255,0.7);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title with Fish Emojis ---
st.markdown('<div class="big-font">🐟🐠 Fish Count Predictor 🐬🦈</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)

# --- Aqua Box for Inputs ---
with st.container():
    st.markdown('<div class="aqua-box">', unsafe_allow_html=True)
    st.markdown("### 🌊 Enter Water Parameters")
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0)
    salinity = st.number_input("🧂 Salinity (PSU)", min_value=0.0, max_value=50.0, value=35.0)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Model Selection ---
models_dir = r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\models"
per_species_dir = os.path.join(models_dir, "per_species")
per_length_dir = os.path.join(models_dir, "per_length")
per_weight_dir = os.path.join(models_dir, "per_weight")

with st.container():
    st.markdown('<div class="aqua-box">', unsafe_allow_html=True)
    st.markdown("### 🐟 Select Prediction Model")
    model_type = st.selectbox("Model Type", ["General", "By Species", "By Length", "By Weight"])

    if model_type == "By Species":
        species_models = [f for f in os.listdir(per_species_dir) if f.endswith(".pkl")]
        species_names = [f.replace("fish_count_model_", "").replace(".pkl", "") for f in species_models]
        species = st.selectbox("Select Species", species_names)
        model_path = os.path.join(per_species_dir, f"fish_count_model_{species}.pkl")
    elif model_type == "By Length":
        length_labels = ["short", "medium", "long"]
        length = st.selectbox("Select Length Category", length_labels)
        model_path = os.path.join(per_length_dir, f"fish_count_model_length_{length}.pkl")
    elif model_type == "By Weight":
        weight_labels = ["light", "medium", "heavy"]
        weight = st.selectbox("Select Weight Category", weight_labels)
        model_path = os.path.join(per_weight_dir, f"fish_count_model_weight_{weight}.pkl")
    else:
        model_path = os.path.join(models_dir, "fish_count_model.pkl")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Button and Output ---
with st.container():
    st.markdown('<div class="aqua-box">', unsafe_allow_html=True)
    if st.button("🐠 Predict Fish Count"):
        model = joblib.load(model_path)
        X = pd.DataFrame([[temperature, salinity]], columns=["Temperature (°C)", "Salinity (PSU)"])
        prediction = model.predict(X)[0]
        st.markdown(f"<h2 style='color:#0e7fa6;'>🐟 Estimated Fish Count: <b>{prediction:.2f}</b></h2>", unsafe_allow_html=True)
        st.balloons()

        # --- Comparison Graphs ---
        st.markdown("### 📊 Comparison with All Categories")
        compare_data = []
        if model_type == "By Species":
            for sp, fname in zip(species_names, species_models):
                m = joblib.load(os.path.join(per_species_dir, fname))
                pred = m.predict(X)[0]
                compare_data.append({"Category": sp, "Predicted Count": pred})
            df_compare = pd.DataFrame(compare_data)
            st.bar_chart(df_compare.set_index("Category"))
        elif model_type == "By Length":
            for l in length_labels:
                mpath = os.path.join(per_length_dir, f"fish_count_model_length_{l}.pkl")
                if os.path.exists(mpath):
                    m = joblib.load(mpath)
                    pred = m.predict(X)[0]
                    compare_data.append({"Category": l, "Predicted Count": pred})
            df_compare = pd.DataFrame(compare_data)
            st.bar_chart(df_compare.set_index("Category"))
        elif model_type == "By Weight":
            for w in weight_labels:
                mpath = os.path.join(per_weight_dir, f"fish_count_model_weight_{w}.pkl")
                if os.path.exists(mpath):
                    m = joblib.load(mpath)
                    pred = m.predict(X)[0]
                    compare_data.append({"Category": w, "Predicted Count": pred})
            df_compare = pd.DataFrame(compare_data)
            st.bar_chart(df_compare.set_index("Category"))
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer with Fish Graphics ---
st.markdown("""
<div style="text-align:center; margin-top:2rem;">
    <span class="fish-emoji">🐟</span>
    <span class="fish-emoji">🐠</span>
    <span class="fish-emoji">🦈</span>
    <span class="fish-emoji">🐬</span>
    <span class="fish-emoji">🐡</span>
</div>
""", unsafe_allow_html=True)

# ...existing code...

# --- Animated Fish GIFs Swimming Across ---
st.markdown("""
<style>
.fish-swim {
  position: fixed;
  top: 20vh;
  left: -200px;
  width: 400px;
  z-index: 9999;
  animation: swim 4s linear infinite;
}
.fish-swim2 {
  position: fixed;
  bottom: 10vh;
  left: -200px;
  width: 100px;
  z-index: 9999;
  animation: swim2 8s ;
}
@keyframes swim {
  0% { left: -200px; }
  100% { left: 110vw; }
}
@keyframes swim2 {
  0% { left: -200px; }
  100% { left: 110vw; }
}
</style>
<img src="https://i.pinimg.com/originals/66/94/b3/6694b3076508b8e9229c378efa5d66d7.gif" class="fish-swim"/>
<img src="https://i.pinimg.com/originals/92/61/b3/9261b3445438ca96b8ecec445171704b.gif" class="fish-swim2"/>
""", unsafe_allow_html=True)

# ...existing Streamlit code...

# Static GIFs in corners
st.markdown("""
<style>
.corner-fish-left {
  position: fixed;
  left: 10px;
  bottom: 10px;
  width: 200px;
  z-index: 9999;
}
.corner-fish-right {
  position: fixed;
  right: 10px;
  bottom: 10px;
  width: 200px;
  z-index: 9999;
}
.fish-swim-rtl {
  position: fixed;
  bottom: 5vh;
  right: -200px;
  width: 120px;
  z-index: 9999;
  animation: swim-rtl 12s ;
}
@keyframes swim-rtl {
   0% { right: -200px; }
  100% { right: 110vw; }
}
.stApp {
    background: url('https://scitechdaily.com/images/Deep-Ocean-Current.gif') no-repeat center center fixed;
    background-size: cover;
}
</style>
<img src="https://i.pinimg.com/originals/0d/5c/98/0d5c98c686e322b58611c7568dcaf0ab.gif" class="corner-fish-left"/>
<img src="https://giffiles.alphacoders.com/201/20164.gif" class="corner-fish-right"/>
<img src="https://media4.giphy.com/media/v1.Y2lkPTZjMDliOTUydHY2bWNqc2p3bW0wMDZsem82d3FyMm9yNnViYnpobzExZnJrbjFxaSZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/cmGcqY1eIS5bQISQrj/200.gif" class="fish-swim-rtl"/>
""", unsafe_allow_html=True)





