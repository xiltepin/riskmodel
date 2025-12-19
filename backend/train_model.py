"""
Insurance Risk Assessment Model Training
Trains Random Forest Classifier on insurance application data
"""
import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import os
from datetime import datetime

# Database connection
DB_CONFIG = {
    'dbname': 'insurance_risk',
    'user': 'insurance_user',
    'password': 'insurance_user',
    'host': 'localhost',
    'port': '5432'
}

# Model configuration
MODEL_CONFIG = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 10,            # Max tree depth
    'min_samples_split': 10,    # Min samples to split node
    'min_samples_leaf': 5,      # Min samples in leaf
    'random_state': 42,         # For reproducibility
    'n_jobs': -1                # Use all CPU cores
}

def load_training_data():
    """Load training data from PostgreSQL"""
    print("📊 Loading training data from database...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Load data using the view we created
        query = """
            SELECT 
                age,
                gender_encoded,
                years_licensed,
                num_claims_3yr,
                total_claim_amount,
                at_fault_claims,
                vehicle_age,
                annual_mileage,
                license_type_encoded,
                marital_encoded,
                prior_insurance_lapses,
                location_risk_score,
                application_denied,
                risk_level
            FROM training_features
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"   ✓ Loaded {len(df)} records")
        print(f"   ✓ Features: {len(df.columns) - 2}")  # -2 for target columns
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None

def prepare_features(df):
    """Prepare features and target variables"""
    print("\n🔧 Preparing features...")
    
    # Feature columns (exclude targets)
    feature_cols = [
        'age', 'gender_encoded', 'years_licensed', 'num_claims_3yr',
        'total_claim_amount', 'at_fault_claims', 'vehicle_age',
        'annual_mileage', 'license_type_encoded', 'marital_encoded',
        'prior_insurance_lapses', 'location_risk_score'
    ]
    
    X = df[feature_cols]
    y_denial = df['application_denied'].astype(int)  # Binary classification
    
    # Map risk levels to numeric
    risk_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'VERY_HIGH': 3}
    y_risk = df['risk_level'].map(risk_map)
    
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Target distribution (denial):")
    print(f"      Approved: {(y_denial == 0).sum()} ({(y_denial == 0).sum()/len(y_denial)*100:.1f}%)")
    print(f"      Denied:   {(y_denial == 1).sum()} ({(y_denial == 1).sum()/len(y_denial)*100:.1f}%)")
    
    return X, y_denial, y_risk, feature_cols

def train_denial_model(X, y, feature_names):
    """Train model to predict application denial"""
    print("\n🌳 Training Random Forest for denial prediction...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of denial
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance:")
    print(f"   ✓ Accuracy: {accuracy*100:.2f}%")
    print(f"   ✓ ROC AUC Score: {roc_auc:.4f}")
    
    # Classification report
    print(f"\n📊 Detailed Metrics:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Approved', 'Denied'],
                                digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"🎯 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Approved  Denied")
    print(f"   Actual Approved  {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"          Denied    {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Feature Importance (Top 5):")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']:25s}: {row['importance']:.4f}")
    
    return model, feature_importance, {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'test_size': len(X_test),
        'train_size': len(X_train)
    }

def train_risk_model(X, y_risk, feature_names):
    """Train model to predict risk level (LOW/MEDIUM/HIGH/VERY_HIGH)"""
    print("\n🌳 Training Random Forest for risk level prediction...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Risk Level Accuracy: {accuracy*100:.2f}%")
    
    return model

def save_models(denial_model, risk_model, feature_names, metrics):
    """Save trained models and metadata"""
    print("\n💾 Saving models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    model_data = {
        'denial_model': denial_model,
        'risk_model': risk_model,
        'feature_names': feature_names,
        'metrics': metrics,
        'model_version': 'v1.0',
        'trained_at': datetime.now().isoformat(),
        'model_config': MODEL_CONFIG
    }
    
    joblib.dump(model_data, 'models/risk_model.pkl')
    
    print(f"   ✓ Models saved to: models/risk_model.pkl")
    print(f"   ✓ Model version: v1.0")
    print(f"   ✓ File size: {os.path.getsize('models/risk_model.pkl') / 1024:.1f} KB")

def test_prediction(model_data):
    """Test model with sample predictions - Japan version"""
    print("\n🧪 Testing model with sample data...")
    
    denial_model = model_data['denial_model']
    risk_model = model_data['risk_model']
    feature_names = model_data['feature_names']
    
    risk_labels = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
    
    # Test case 1: Gold license - Excellent driver (should be LOW risk)
    test_case_1 = pd.DataFrame([{
        'age': 45,
        'gender_encoded': 0,  # Female
        'years_licensed': 25,
        'num_claims_3yr': 0,
        'total_claim_amount': 0,
        'at_fault_claims': 0,
        'vehicle_age': 4,
        'annual_mileage': 9000,
        'license_type_encoded': 0,  # Gold = 0
        'marital_encoded': 0,  # Married
        'prior_insurance_lapses': 0,
        'location_risk_score': 0.3
    }], columns=feature_names)
    
    denial_prob_1 = denial_model.predict_proba(test_case_1)[0][1]
    risk_pred_1 = risk_model.predict(test_case_1)[0]
    
    print(f"\n   Test Case 1 - Gold License (優良運転者):")
    print(f"   Age: 45, 25 years licensed, 0 claims, Gold license")
    print(f"   → Denial Probability: {denial_prob_1*100:.1f}%")
    print(f"   → Risk Level: {risk_labels[risk_pred_1]}")
    print(f"   → Decision: {'✅ APPROVED' if denial_prob_1 < 0.5 else '❌ DENIED'}")
    
    # Test case 2: Green license - Beginner high risk
    test_case_2 = pd.DataFrame([{
        'age': 18,
        'gender_encoded': 1,  # Male
        'years_licensed': 0,
        'num_claims_3yr': 2,
        'total_claim_amount': 18000,
        'at_fault_claims': 2,
        'vehicle_age': 2,
        'annual_mileage': 15000,
        'license_type_encoded': 2,  # Green = 2
        'marital_encoded': 1,  # Single
        'prior_insurance_lapses': 1,
        'location_risk_score': 0.6
    }], columns=feature_names)
    
    denial_prob_2 = denial_model.predict_proba(test_case_2)[0][1]
    risk_pred_2 = risk_model.predict(test_case_2)[0]
    
    print(f"\n   Test Case 2 - Green License (初心者):")
    print(f"   Age: 18, <1 year licensed, 2 claims, Green license")
    print(f"   → Denial Probability: {denial_prob_2*100:.1f}%")
    print(f"   → Risk Level: {risk_labels[risk_pred_2]}")
    print(f"   → Decision: {'✅ APPROVED' if denial_prob_2 < 0.5 else '❌ DENIED'}")

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("  🚗 Insurance Risk Assessment Model Training")
    print("=" * 70)
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    # Prepare features
    X, y_denial, y_risk, feature_names = prepare_features(df)
    
    # Train denial prediction model
    denial_model, feature_importance, metrics = train_denial_model(
        X, y_denial, feature_names
    )
    
    # Train risk level prediction model
    risk_model = train_risk_model(X, y_risk, feature_names)
    
    # Save models
    model_data = {
        'denial_model': denial_model,
        'risk_model': risk_model,
        'feature_names': feature_names,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_version': 'v1.0',
        'trained_at': datetime.now().isoformat(),
        'model_config': MODEL_CONFIG
    }
    
    save_models(denial_model, risk_model, feature_names, metrics)
    
    # Test predictions
    test_prediction(model_data)
    
    print("\n" + "=" * 70)
    print("  ✅ Training Complete!")
    print("=" * 70)
    print(f"\n📦 Next steps:")
    print(f"   1. Review model metrics above")
    print(f"   2. Check models/risk_model.pkl file")
    print(f"   3. Build Flask API to serve predictions")
    print(f"   4. Integrate with Angular frontend")

if __name__ == "__main__":
    main()