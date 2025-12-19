# Insurance Risk Assessment POC - Update (Dec 19, 2025)

## Project Overview
AI-powered auto insurance risk evaluation system.  
Uses synthetic data + Random Forest model to predict application denial and risk level.

## Objective
Build a full-stack Proof of Concept:  
- Generate realistic training data (PostgreSQL)  
- Train ML model (Random Forest)  
- Flask API backend (predictions + logging)  
- Modern Angular frontend (Material Design 3, standalone components)

## Current Status
- Data: 1000 records generated, denial rate ~23%
- Model: Trained v1.0 (94% accuracy, 0.991 AUC)
- Backend: Flask API running on http://localhost:5001 (CORS enabled)
- Frontend: Angular 18+ standalone app running on http://localhost:4300
- Integration: Frontend successfully calls `/api/predict`

## Working Features
- Health, model-info, predict endpoints
- Form with all required fields (English UI)
- Real-time prediction with colored results (LOW/MEDIUM/HIGH/VERY_HIGH)
- Premium multiplier and confidence displayed

## Current Issues to Solve
1. **Spinner loops forever** when clicking "Evaluar Riesgo"  
   → Prediction works (backend returns 200), but frontend not updating `result`  
   → Likely cause: Angular not detecting response change or HTTP subscription issue

2. **Database logging error**  
