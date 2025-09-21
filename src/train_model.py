import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv(r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\data\fish_data.csv")

# General model (keep this)
X = df[["Temperature (°C)", "Salinity (PSU)"]]
y = df["Count"]
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\models\fish_count_model.pkl")
print("✅ General model trained and saved as models/fish_count_model.pkl")

# Per-species models
species_list = df["Species"].unique()
per_species_dir = r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\models\per_species"
os.makedirs(per_species_dir, exist_ok=True)

for species in species_list:
    df_species = df[df["Species"] == species]
    X_species = df_species[["Temperature (°C)", "Salinity (PSU)"]]
    y_species = df_species["Count"]
    model_species = LinearRegression()
    model_species.fit(X_species, y_species)
    model_path = os.path.join(per_species_dir, f"fish_count_model_{species}.pkl")
    joblib.dump(model_species, model_path)
    print(f"✅ Model trained and saved for {species}: {model_path}")

    # ...existing code...

# Define bins and labels
length_bins = [0, 20, 40, float('inf')]
length_labels = ['short', 'medium', 'long']
weight_bins = [0, 500, 2000, float('inf')]
weight_labels = ['light', 'medium', 'heavy']

# Add columns for categories
df['LengthCategory'] = pd.cut(df['Fish Length (cm)'], bins=length_bins, labels=length_labels, right=False)
df['WeightCategory'] = pd.cut(df['Weight (g)'], bins=weight_bins, labels=weight_labels, right=False)

# Per-length models
per_length_dir = r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\models\per_length"
os.makedirs(per_length_dir, exist_ok=True)

for length in length_labels:
    df_length = df[df['LengthCategory'] == length]
    if df_length.empty:
        continue
    X_length = df_length[["Temperature (°C)", "Salinity (PSU)"]]
    y_length = df_length["Count"]
    model_length = LinearRegression()
    model_length.fit(X_length, y_length)
    model_path = os.path.join(per_length_dir, f"fish_count_model_length_{length}.pkl")
    joblib.dump(model_length, model_path)
    print(f"✅ Model trained and saved for length '{length}': {model_path}")

# Per-weight models
per_weight_dir = r"C:\Users\aa\Desktop\Coding\sih-marine-prototype\models\per_weight"
os.makedirs(per_weight_dir, exist_ok=True)

for weight in weight_labels:
    df_weight = df[df['WeightCategory'] == weight]
    if df_weight.empty:
        continue
    X_weight = df_weight[["Temperature (°C)", "Salinity (PSU)"]]
    y_weight = df_weight["Count"]
    model_weight = LinearRegression()
    model_weight.fit(X_weight, y_weight)
    model_path = os.path.join(per_weight_dir, f"fish_count_model_weight_{weight}.pkl")
    joblib.dump(model_weight, model_path)
    print(f"✅ Model trained and saved for weight '{weight}': {model_path}")
    