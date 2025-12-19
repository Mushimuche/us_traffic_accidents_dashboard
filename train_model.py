import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Configuration
training_file_path = "us_accidents_ca_balanced.csv"
model_save_path = "accident_model_data.joblib"

def train_and_save():
    print(f"Loading training data from {training_file_path}...")
    
    if not os.path.exists(training_file_path):
        print(f"Error: {training_file_path} not found.")
        return

    # 1. Load Data (Only columns needed for training)
    train_cols = ['Severity', 'Start_Time', 'Weather_Condition', 'Temperature(F)', 
                  'Humidity(%)', 'Traffic_Signal', 'Junction', 'Crossing']
    
    df_train = pd.read_csv(training_file_path, usecols=lambda c: c in train_cols)
    
    # 2. Preprocessing
    print("Preprocessing...")
    df_train['Start_Time'] = pd.to_datetime(df_train['Start_Time'], errors='coerce')
    df_train['Hour'] = df_train['Start_Time'].dt.hour
    
    # Imputation
    df_train['Weather_Condition'] = df_train['Weather_Condition'].fillna('Clear')
    df_train['Temperature(F)'] = df_train['Temperature(F)'].fillna(df_train['Temperature(F)'].median())
    df_train['Humidity(%)'] = df_train['Humidity(%)'].fillna(df_train['Humidity(%)'].median())
    
    # Encoding
    le_weather = LabelEncoder()
    df_train['Weather_Encoded'] = le_weather.fit_transform(df_train['Weather_Condition'].astype(str))
    
    # 3. Train
    X = df_train[['Hour', 'Weather_Encoded', 'Temperature(F)', 'Humidity(%)', 
                  'Traffic_Signal', 'Junction', 'Crossing']]
    y = df_train['Severity']
    
    print("Training Random Forest (this may take a minute)...")
    # Using compression to keep file size small for GitHub
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    # 4. Save
    print(f"Saving to {model_save_path}...")
    joblib.dump({'model': rf_model, 'encoder': le_weather}, model_save_path, compress=3)
    
    print("Success! Now git commit and push 'accident_model_data.joblib' to your repo.")

if __name__ == "__main__":
    train_and_save()