import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_dir, "datasets", "accidents.csv")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Standardize target
    # In accidents.csv, labels might be 'Slight Injury', 'Serious Injury', 'Fatal injury' (lowercase 'injury')
    df['Accident_severity'] = df['Accident_severity'].astype(str).str.strip().str.title()
    target_mapping = {
        "Fatal Injury": 0,
        "Serious Injury": 1,
        "Slight Injury": 2
    }
    y = df['Accident_severity'].map(target_mapping)
    
    # If any row couldn't be mapped, drop it
    valid_indices = y.dropna().index
    df = df.loc[valid_indices]
    y = y.dropna().astype(int)
    
    # 6 Core features used in the Streamlit app
    feature_cols = [
        "Day_of_week",
        "Age_band_of_driver",
        "Light_conditions",
        "Weather_conditions",
        "Number_of_vehicles_involved",
        "Number_of_casualties"
    ]
    
    X = df[feature_cols].copy()
    
    encoders = {}
    column_modes = {}
    
    # Process each feature
    for col in feature_cols:
        # Save the mode for missing value filling
        column_modes[col] = X[col].mode()[0] if not X[col].dropna().empty else 0
        X[col] = X[col].fillna(column_modes[col])
        
        # Fit LabelEncoder if categorical
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            # Convert values to string to avoid comparison issues
            X_clean = X[col].astype(str)
            le.fit(X_clean)
            X[col] = le.transform(X_clean)
            encoders[col] = le
        else:
            X[col] = X[col].astype(float)
            
    print(f"Features: {feature_cols}")
    print(f"Feature modes: {column_modes}")
    
    # Split into train and test to monitor metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Train RandomForestClassifier with balanced class weights to resolve class imbalance
    print("Training Random Forest Classifier with balanced class weights...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=["Fatal Injury", "Serious Injury", "Slight Injury"]))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the updated model and its encoders/modes metadata
    model_path = os.path.join(project_dir, "models", "accident_model.pkl")
    metadata_path = os.path.join(project_dir, "models", "accident_model_metadata.pkl")
    
    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving metadata to {metadata_path}...")
    metadata = {
        "encoders": encoders,
        "column_modes": column_modes,
        "feature_cols": feature_cols
    }
    joblib.dump(metadata, metadata_path)
    
    print("Model retraining and metadata serialization complete!")

if __name__ == "__main__":
    main()
