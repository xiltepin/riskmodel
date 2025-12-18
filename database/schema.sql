-- Insurance Risk Assessment Database Schema
-- Run this after creating the database: CREATE DATABASE insurance_risk;

-- Training data table (historical insurance applications)
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age BETWEEN 16 AND 100),
    gender VARCHAR(10) NOT NULL,
    years_licensed INTEGER NOT NULL CHECK (years_licensed >= 0),
    num_claims_3yr INTEGER NOT NULL DEFAULT 0,
    total_claim_amount DECIMAL(10, 2) NOT NULL DEFAULT 0,
    at_fault_claims INTEGER NOT NULL DEFAULT 0,
    vehicle_age INTEGER NOT NULL CHECK (vehicle_age >= 0),
    annual_mileage INTEGER NOT NULL CHECK (annual_mileage >= 0),
    credit_score INTEGER CHECK (credit_score BETWEEN 300 AND 850),
    marital_status VARCHAR(20) NOT NULL,
    prior_insurance_lapses INTEGER NOT NULL DEFAULT 0,
    location_risk_score DECIMAL(3, 2) NOT NULL CHECK (location_risk_score BETWEEN 0 AND 1),
    -- Target variables
    application_denied BOOLEAN NOT NULL,
    risk_level VARCHAR(20), -- LOW, MEDIUM, HIGH, VERY_HIGH
    premium_multiplier DECIMAL(4, 2), -- 0.8 to 3.0 (80% to 300% of base rate)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction logs (store API predictions for model improvement)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    age INTEGER NOT NULL,
    gender VARCHAR(10),
    years_licensed INTEGER,
    num_claims_3yr INTEGER,
    total_claim_amount DECIMAL(10, 2),
    at_fault_claims INTEGER,
    vehicle_age INTEGER,
    annual_mileage INTEGER,
    credit_score INTEGER,
    marital_status VARCHAR(20),
    prior_insurance_lapses INTEGER,
    location_risk_score DECIMAL(3, 2),
    -- Model outputs
    predicted_denial_probability DECIMAL(5, 4),
    predicted_risk_level VARCHAR(20),
    model_confidence DECIMAL(5, 4),
    model_version VARCHAR(20) DEFAULT 'v1.0',
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_training_denied ON training_data(application_denied);
CREATE INDEX IF NOT EXISTS idx_training_risk ON training_data(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(predicted_at);

-- View for model training features
CREATE OR REPLACE VIEW training_features AS
SELECT 
    age,
    CASE 
        WHEN gender = 'Male' THEN 1
        WHEN gender = 'Female' THEN 0
        ELSE 0.5
    END as gender_encoded,
    years_licensed,
    num_claims_3yr,
    total_claim_amount,
    at_fault_claims,
    vehicle_age,
    annual_mileage,
    credit_score,
    CASE marital_status
        WHEN 'Married' THEN 0
        WHEN 'Single' THEN 1
        WHEN 'Divorced' THEN 2
        ELSE 3
    END as marital_encoded,
    prior_insurance_lapses,
    location_risk_score,
    application_denied,
    risk_level
FROM training_data;
