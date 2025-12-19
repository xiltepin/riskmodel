"""
Insurance Risk Assessment API
Flask REST API for ML model predictions
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import psycopg2
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Angular frontend

# Database configuration
DB_CONFIG = {
    'dbname': 'insurance_risk',
    'user': 'insurance_user',
    'password': 'insurance_user',
    'host': 'localhost',
    'port': '5432'
}

# Load model on startup
MODEL_PATH = 'models/risk_model.pkl'
model_data = None

def load_model():
    """Load trained model from disk"""
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully!")
        print(f"   Version: {model_data['model_version']}")
        print(f"   Trained: {model_data['trained_at']}")
        print(f"   Accuracy: {model_data['metrics']['accuracy']*100:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model when app starts
if not load_model():
    print("⚠️  Warning: Model not loaded. API will not work properly.")

# Feature name mapping for user-friendly input
FEATURE_MAPPING = {
    'age': 'age',
    'gender': 'gender_encoded',
    'years_licensed': 'years_licensed',
    'num_claims_3yr': 'num_claims_3yr',
    'total_claim_amount': 'total_claim_amount',
    'at_fault_claims': 'at_fault_claims',
    'vehicle_age': 'vehicle_age',
    'annual_mileage': 'annual_mileage',
    'credit_score': 'credit_score',
    'marital_status': 'marital_encoded',
    'prior_insurance_lapses': 'prior_insurance_lapses',
    'location_risk_score': 'location_risk_score'
}

def encode_gender(gender):
    """Encode gender to numeric value"""
    gender_map = {
        'male': 1,
        'female': 0,
        'other': 0.5,
        'm': 1,
        'f': 0
    }
    return gender_map.get(gender.lower(), 0.5)

def encode_marital_status(status):
    """Encode marital status to numeric value"""
    marital_map = {
        'married': 0,
        'single': 1,
        'divorced': 2,
        'widowed': 3
    }
    return marital_map.get(status.lower(), 1)

def prepare_features(data):
    """Prepare features from input data"""
    try:
        # Encode categorical variables
        gender_encoded = encode_gender(data.get('gender', 'other'))
        marital_encoded = encode_marital_status(data.get('marital_status', 'single'))
        
        # Build feature array in correct order
        features = [
            float(data.get('age', 30)),
            gender_encoded,
            float(data.get('years_licensed', 5)),
            float(data.get('num_claims_3yr', 0)),
            float(data.get('total_claim_amount', 0)),
            float(data.get('at_fault_claims', 0)),
            float(data.get('vehicle_age', 5)),
            float(data.get('annual_mileage', 12000)),
            float(data.get('credit_score', 680)),
            marital_encoded,
            float(data.get('prior_insurance_lapses', 0)),
            float(data.get('location_risk_score', 0.5))
        ]
        
        return np.array(features).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error preparing features: {str(e)}")

def log_prediction(input_data, prediction_result):
    """Log prediction to database for model improvement - FIXED VERSION"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # CONVERTIR TODOS LOS VALORES A TIPOS NATIVOS DE PYTHON
        # Esto evita el error "schema np does not exist"
        safe_data = {
            'age': int(input_data.get('age', 0)) if input_data.get('age') is not None else None,
            'gender': str(input_data.get('gender', '')),
            'years_licensed': int(input_data.get('years_licensed', 0)) if input_data.get('years_licensed') is not None else None,
            'num_claims_3yr': int(input_data.get('num_claims_3yr', 0)),
            'total_claim_amount': float(input_data.get('total_claim_amount', 0.0)),
            'at_fault_claims': int(input_data.get('at_fault_claims', 0)),
            'vehicle_age': int(input_data.get('vehicle_age', 0)),
            'annual_mileage': int(input_data.get('annual_mileage', 0)),
            'credit_score': int(input_data.get('credit_score', 0)),
            'marital_status': str(input_data.get('marital_status', '')),
            'prior_insurance_lapses': int(input_data.get('prior_insurance_lapses', 0)),
            'location_risk_score': float(input_data.get('location_risk_score', 0.5)),
            'predicted_denial_probability': float(prediction_result['denial_probability']),
            'predicted_risk_level': str(prediction_result['risk_level']),
            'model_confidence': float(prediction_result['confidence']),
            'model_version': str(prediction_result.get('model_version', 'v1.0'))
        }
        
        cur.execute("""
            INSERT INTO predictions (
                age, gender, years_licensed, num_claims_3yr,
                total_claim_amount, at_fault_claims, vehicle_age,
                annual_mileage, credit_score, marital_status,
                prior_insurance_lapses, location_risk_score,
                predicted_denial_probability, predicted_risk_level,
                model_confidence, model_version
            ) VALUES (
                %(age)s, %(gender)s, %(years_licensed)s, %(num_claims_3yr)s,
                %(total_claim_amount)s, %(at_fault_claims)s, %(vehicle_age)s,
                %(annual_mileage)s, %(credit_score)s, %(marital_status)s,
                %(prior_insurance_lapses)s, %(location_risk_score)s,
                %(predicted_denial_probability)s, %(predicted_risk_level)s,
                %(model_confidence)s, %(model_version)s
            )
        """, safe_data)
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("✅ Prediction logged successfully to database")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not log prediction: {e}")

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Insurance Risk Assessment API',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'model_info': '/api/model-info',
            'predict': '/api/predict (POST)',
            'stats': '/api/stats'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = model_data is not None
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_version': model_data['model_version'],
        'trained_at': model_data['trained_at'],
        'accuracy': round(model_data['metrics']['accuracy'] * 100, 2),
        'roc_auc': round(model_data['metrics']['roc_auc'], 4),
        'features': model_data['feature_names'],
        'feature_count': len(model_data['feature_names'])
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict insurance application outcome
    
    Expected JSON input:
    {
        "age": 28,
        "gender": "male",
        "years_licensed": 8,
        "num_claims_3yr": 1,
        "total_claim_amount": 4500,
        "at_fault_claims": 0,
        "vehicle_age": 3,
        "annual_mileage": 15000,
        "credit_score": 720,
        "marital_status": "single",
        "prior_insurance_lapses": 0,
        "location_risk_score": 0.45
    }
    """
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'years_licensed']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Get models
        denial_model = model_data['denial_model']
        risk_model = model_data['risk_model']
        
        # Make predictions
        denial_proba = denial_model.predict_proba(features)[0]
        denial_prob = denial_proba[1]  # Probability of denial
        predicted_denial = denial_prob >= 0.5
        
        # Confidence is the max probability
        confidence = max(denial_proba)
        
        # Predict risk level
        risk_pred = risk_model.predict(features)[0]
        risk_labels = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
        risk_level = risk_labels[risk_pred]
        
        # Calculate premium multiplier based on risk
        premium_multipliers = {
            'LOW': round(np.random.uniform(0.8, 1.1), 2),
            'MEDIUM': round(np.random.uniform(1.1, 1.5), 2),
            'HIGH': round(np.random.uniform(1.5, 2.2), 2),
            'VERY_HIGH': round(np.random.uniform(2.2, 3.0), 2)
        }
        premium_multiplier = premium_multipliers[risk_level]
        
        # Build response
        result = {
            'denial_probability': round(denial_prob, 4),
            'predicted_denial': bool(predicted_denial),
            'risk_level': risk_level,
            'confidence': round(confidence, 4),
            'premium_multiplier': premium_multiplier,
            'decision': 'DENIED' if predicted_denial else 'APPROVED',
            'model_version': model_data['model_version'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction to database
        log_prediction(data, result)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics from database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Total predictions
        cur.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cur.fetchone()[0]
        
        # Average denial probability
        cur.execute("SELECT AVG(predicted_denial_probability) FROM predictions")
        avg_denial_prob = cur.fetchone()[0]
        
        # Risk level distribution
        cur.execute("""
            SELECT predicted_risk_level, COUNT(*) 
            FROM predictions 
            GROUP BY predicted_risk_level
            ORDER BY predicted_risk_level
        """)
        risk_distribution = {row[0]: row[1] for row in cur.fetchall()}
        
        # Recent predictions (last 10)
        cur.execute("""
            SELECT age, predicted_risk_level, predicted_denial_probability, predicted_at
            FROM predictions 
            ORDER BY predicted_at DESC 
            LIMIT 10
        """)
        recent = [
            {
                'age': row[0],
                'risk_level': row[1],
                'denial_probability': float(row[2]),
                'timestamp': row[3].isoformat()
            }
            for row in cur.fetchall()
        ]
        
        cur.close()
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'average_denial_probability': round(float(avg_denial_prob or 0), 4),
            'risk_distribution': risk_distribution,
            'recent_predictions': recent
        })
        
    except Exception as e:
        return jsonify({'error': f'Could not fetch stats: {str(e)}'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple applications at once"""
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        applications = data.get('applications', [])
        
        if not applications:
            return jsonify({'error': 'No applications provided'}), 400
        
        results = []
        for app_data in applications:
            features = prepare_features(app_data)
            
            denial_proba = model_data['denial_model'].predict_proba(features)[0]
            denial_prob = denial_proba[1]
            
            risk_pred = model_data['risk_model'].predict(features)[0]
            risk_labels = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
            
            results.append({
                'denial_probability': round(denial_prob, 4),
                'predicted_denial': bool(denial_prob >= 0.5),
                'risk_level': risk_labels[risk_pred],
                'confidence': round(max(denial_proba), 4)
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("  🚗 Insurance Risk Assessment API")
    print("=" * 70)
    print(f"\n🌐 Starting server on http://localhost:5001")
    print(f"📡 CORS enabled for Angular frontend")
    print(f"📊 Model version: {model_data['model_version'] if model_data else 'N/A'}")
    print(f"\n🔗 Available endpoints:")
    print(f"   GET  /api/health       - Health check")
    print(f"   GET  /api/model-info   - Model information")
    print(f"   POST /api/predict      - Single prediction")
    print(f"   POST /api/batch-predict - Multiple predictions")
    print(f"   GET  /api/stats        - Prediction statistics")
    print("\n" + "=" * 70)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
