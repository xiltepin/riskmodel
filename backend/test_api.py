"""
API Testing Script
Test the Flask API endpoints
"""
import requests
import json

BASE_URL = "http://localhost:5001"

def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/api/health")
    print_response("🏥 Health Check", response)

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/api/model-info")
    print_response("📊 Model Information", response)

def test_predict_safe_driver():
    """Test prediction for safe driver"""
    data = {
        "age": 35,
        "gender": "female",
        "years_licensed": 15,
        "num_claims_3yr": 0,
        "total_claim_amount": 0,
        "at_fault_claims": 0,
        "vehicle_age": 3,
        "annual_mileage": 12000,
        "credit_score": 750,
        "marital_status": "married",
        "prior_insurance_lapses": 0,
        "location_risk_score": 0.3
    }
    
    response = requests.post(f"{BASE_URL}/api/predict", json=data)
    print_response("✅ Safe Driver Prediction", response)

def test_predict_risky_driver():
    """Test prediction for risky driver"""
    data = {
        "age": 21,
        "gender": "male",
        "years_licensed": 2,
        "num_claims_3yr": 3,
        "total_claim_amount": 25000,
        "at_fault_claims": 2,
        "vehicle_age": 1,
        "annual_mileage": 20000,
        "credit_score": 580,
        "marital_status": "single",
        "prior_insurance_lapses": 1,
        "location_risk_score": 0.75
    }
    
    response = requests.post(f"{BASE_URL}/api/predict", json=data)
    print_response("❌ Risky Driver Prediction", response)

def test_stats():
    """Test statistics endpoint"""
    response = requests.get(f"{BASE_URL}/api/stats")
    print_response("📈 Prediction Statistics", response)

def test_batch_predict():
    """Test batch prediction"""
    data = {
        "applications": [
            {
                "age": 28,
                "gender": "male",
                "years_licensed": 8,
                "num_claims_3yr": 1,
                "credit_score": 720
            },
            {
                "age": 45,
                "gender": "female",
                "years_licensed": 20,
                "num_claims_3yr": 0,
                "credit_score": 800
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/api/batch-predict", json=data)
    print_response("🔢 Batch Prediction", response)

if __name__ == "__main__":
    print("=" * 70)
    print("  🧪 API Testing Suite")
    print("=" * 70)
    print("\n⚠️  Make sure the Flask API is running on port 5001!")
    print("   Run: python3 app.py")
    
    input("\nPress Enter to start tests...")
    
    try:
        test_health()
        test_model_info()
        test_predict_safe_driver()
        test_predict_risky_driver()
        test_batch_predict()
        test_stats()
        
        print("\n" + "=" * 70)
        print("  ✅ All tests completed!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure Flask is running: python3 app.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
