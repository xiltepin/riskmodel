# 🚗 Insurance Risk Assessment - Backend

Flask API with ML model for auto insurance risk prediction.

---

## 📋 Project Structure

```
backend/
├── .venv/                  # Virtual environment
├── generate_data.py        # Dummy training data generator
├── train_model.py          # ML model training script (coming soon)
├── app.py                  # Flask API server (coming soon)
├── requirements.txt        # Python dependencies
├── models/
│   └── risk_model.pkl      # Trained model (generated after training)
└── README.md              # This file
```

---

## 🛠️ Tech Stack

- **Python 3.12+**
- **Flask** - REST API framework (Port 5001)
- **PostgreSQL** - Training data storage
- **scikit-learn** - Machine learning library
- **psycopg2** - PostgreSQL adapter

---

## 🚀 Setup Instructions

### 1. Prerequisites

```bash
# Ensure PostgreSQL is running
sudo service postgresql status

# Should see: postgresql is running
```

### 2. Create Virtual Environment

```bash
cd ~/tools/chamba/aig/riskmodel/backend
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Current dependencies:**
```
psycopg2-binary==2.9.11
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
flask==3.0.0
flask-cors==4.0.0
joblib==1.3.2
```

### 4. Database Setup

**Database credentials:**
- Database: `insurance_risk`
- User: `insurance_user`
- Password: `insurance_user`
- Host: `localhost`
- Port: `5432`

**Connection config in Python:**
```python
DB_CONFIG = {
    'dbname': 'insurance_risk',
    'user': 'insurance_user',
    'password': 'insurance_user',
    'host': 'localhost',
    'port': '5432'
}
```

### 5. Generate Training Data

```bash
python3 generate_data.py
```

**Expected output:**
- ✅ 1,000 realistic insurance application records
- 📊 ~25-30% denial rate
- 📈 Distribution across LOW, MEDIUM, HIGH, VERY_HIGH risk levels

**Verify data:**
```bash
psql -U insurance_user -d insurance_risk -h localhost -c "SELECT COUNT(*) FROM training_data;"
```

---

## 🧠 Machine Learning Model

### Features Used for Prediction

The model analyzes 12 key factors:

1. **Age** (16-100 years)
2. **Gender** (Male/Female/Other)
3. **Years Licensed** (driving experience)
4. **Number of Claims (3yr)** - total claims in last 3 years
5. **Total Claim Amount ($)** - sum of all claim payouts
6. **At-Fault Claims** - accidents where driver was responsible
7. **Vehicle Age** - age of insured vehicle
8. **Annual Mileage** - yearly driving distance
9. **Credit Score** (300-850)
10. **Marital Status** (Single/Married/Divorced)
11. **Prior Insurance Lapses** - gaps in coverage history
12. **Location Risk Score** (0.0-1.0) - geographic risk factor

### Model Outputs

- **Denial Probability** (0.0-1.0) - likelihood application will be rejected
- **Risk Level** (LOW/MEDIUM/HIGH/VERY_HIGH)
- **Confidence Score** (0.0-1.0) - model's certainty in prediction
- **Premium Multiplier** (0.8-3.0) - recommended rate adjustment

---

## 📊 Training Data Statistics

**Generated with realistic correlations:**

- Young drivers (< 25) → Higher claim rates
- Multiple at-fault accidents → Likely denial
- Poor credit (< 600) → Higher risk classification
- Married drivers → Statistically lower risk
- High mileage (> 20k/year) → Increased exposure

**Risk calculation considers:**
- Age curve (U-shaped: young and elderly are riskier)
- Experience level (years licensed vs age)
- Claims frequency and severity
- Credit score bands
- Combined risk factors

---

## 🔧 Scripts

### `generate_data.py`

Generates 1,000 correlated training records with realistic insurance patterns.

**Usage:**
```bash
python3 generate_data.py
```

**To regenerate data:**
```bash
# Clear existing data
psql -U insurance_user -d insurance_risk -h localhost -c "TRUNCATE TABLE training_data RESTART IDENTITY CASCADE;"

# Generate fresh data
python3 generate_data.py
```

### `train_model.py` *(Coming Soon)*

Trains Random Forest Classifier on training data and saves model.

**Usage:**
```bash
python3 train_model.py
```

**Output:**
- Saved model: `models/risk_model.pkl`
- Training metrics (accuracy, precision, recall)
- Feature importance analysis

### `app.py` *(Coming Soon)*

Flask REST API server.

**Usage:**
```bash
python3 app.py
# Runs on http://localhost:5001
```

**Endpoints:**
- `POST /api/predict` - Get risk assessment for new driver
- `GET /api/health` - API health check
- `GET /api/model-info` - Model version and metrics

---

## 🧪 Testing

### Quick Database Check

```bash
# View sample records
psql -U insurance_user -d insurance_risk -h localhost -c "
  SELECT age, gender, num_claims_3yr, risk_level, application_denied 
  FROM training_data 
  LIMIT 5;
"

# Check denial rate
psql -U insurance_user -d insurance_risk -h localhost -c "
  SELECT 
    application_denied, 
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM training_data), 1) as percentage
  FROM training_data 
  GROUP BY application_denied;
"
```

### Test Python Connection

```python
import psycopg2

conn = psycopg2.connect(
    dbname='insurance_risk',
    user='insurance_user',
    password='insurance_user',
    host='localhost',
    port='5432'
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM training_data;")
print(f"Records: {cur.fetchone()[0]}")
conn.close()
```

---

## 🐛 Troubleshooting

### "Module not found: psycopg2"

```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall
pip install psycopg2-binary
```

### "Password authentication failed"

```bash
# Verify user exists
sudo -u postgres psql -c "\du"

# Reset password
sudo -u postgres psql -c "ALTER USER insurance_user PASSWORD 'insurance_user';"

# Check pg_hba.conf has md5 auth for insurance_user
sudo nano /etc/postgresql/16/main/pg_hba.conf
```

### "Connection refused"

```bash
# Start PostgreSQL
sudo service postgresql start

# Check status
sudo service postgresql status
```

### Virtual Environment Issues

```bash
# Deactivate current venv
deactivate

# Remove and recreate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📝 Development Notes

### Phase 1: Data Generation ✅
- [x] PostgreSQL setup
- [x] Schema creation
- [x] Dummy data generator with realistic correlations
- [x] 1,000 training records generated

### Phase 2: Model Training 🚧
- [ ] Feature engineering
- [ ] Train Random Forest Classifier
- [ ] Model evaluation and metrics
- [ ] Save trained model as .pkl

### Phase 3: Flask API 🚧
- [ ] REST API endpoints
- [ ] Model inference
- [ ] Prediction logging
- [ ] CORS configuration for Angular

### Phase 4: Integration 🚧
- [ ] Connect to Angular frontend
- [ ] API testing
- [ ] Error handling
- [ ] Documentation

---

## 🔐 Security Notes

⚠️ **This is a POC - NOT production-ready!**

**Current setup:**
- Simple password (`insurance_user`)
- No API authentication
- No input validation
- Local database only

**For production, implement:**
- Strong passwords / password manager
- JWT authentication
- Input sanitization
- Rate limiting
- HTTPS only
- Database encryption
- Separate dev/prod environments

---

## 📊 Performance

**Current specs:**
- Training data: 1,000 records (~200KB)
- Expected model size: ~5-10MB
- Training time: ~2-5 seconds
- Inference time: <10ms per prediction

**System requirements:**
- RAM: 2GB minimum (POC running on 80GB system)
- CPU: Any modern multi-core (6-core Ryzen 5 5600G)
- Storage: <100MB for full project

---

## 📞 API Documentation (Coming Soon)

### POST /api/predict

**Request:**
```json
{
  "age": 28,
  "gender": "Male",
  "years_licensed": 8,
  "num_claims_3yr": 1,
  "total_claim_amount": 4500.00,
  "at_fault_claims": 0,
  "vehicle_age": 3,
  "annual_mileage": 15000,
  "credit_score": 720,
  "marital_status": "Single",
  "prior_insurance_lapses": 0,
  "location_risk_score": 0.45
}
```

**Response:**
```json
{
  "denial_probability": 0.23,
  "predicted_denial": false,
  "risk_level": "MEDIUM",
  "confidence": 0.87,
  "premium_multiplier": 1.35,
  "model_version": "v1.0"
}
```

---

## 🎯 Next Steps

1. ✅ Database and data generation complete
2. 🔄 Train ML model
3. 🔄 Build Flask API
4. 🔄 Connect to Angular frontend
5. 🔄 End-to-end testing

---

## 📚 Useful Commands

```bash
# Activate venv
source .venv/bin/activate

# Deactivate venv
deactivate

# Update dependencies
pip freeze > requirements.txt

# Check PostgreSQL status
sudo service postgresql status

# Access database CLI
psql -U insurance_user -d insurance_risk -h localhost

# View logs
tail -f /var/log/postgresql/postgresql-16-main.log

# Check Python version
python3 --version
```

---

## 🤝 Contributing

This is a POC project for learning purposes.

**Project Owner:** xiltepin  
**Environment:** WSL Ubuntu on Windows 11  
**Last Updated:** December 18, 2025

---

## 📄 License

POC / Educational Use Only