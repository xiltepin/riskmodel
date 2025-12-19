"""
Insurance Training Data Generator
Generates 1,000 realistic insurance application records with correlations
"""
import random
import psycopg2
from datetime import datetime, timedelta

# Database connection parameters
DB_CONFIG = {
    'dbname': 'insurance_risk',
    'user': 'insurance_user',
    'password': 'insurance_user',
    'host': 'localhost',
    'port': '5432'
}

def calculate_risk_profile(data):
    """
    Calculate if application should be denied and risk level
    Based on realistic insurance underwriting rules
    """
    risk_score = 0
    
    # Age risk (U-shaped curve: young and very old are riskier)
    if data['age'] < 25:
        risk_score += (25 - data['age']) * 0.5
    elif data['age'] > 70:
        risk_score += (data['age'] - 70) * 0.3
    
    # Experience matters
    if data['years_licensed'] < 3:
        risk_score += 5
    elif data['years_licensed'] < 5:
        risk_score += 2
    
    # Claims history (biggest factor!)
    risk_score += data['num_claims_3yr'] * 8
    risk_score += data['at_fault_claims'] * 5
    
    # Claim amounts
    if data['total_claim_amount'] > 50000:
        risk_score += 10
    elif data['total_claim_amount'] > 20000:
        risk_score += 5
    
    # License type risk (Japan-specific)
    if data['license_type'] == 'green':
        risk_score += 8  # High risk for beginners
    elif data['license_type'] == 'blue':
        risk_score += 3  # Medium risk if violations
    # Gold = 0 (low risk)
    
    # Vehicle age (very old or very new can be problematic)
    if data['vehicle_age'] > 15:
        risk_score += 2
    
    # High mileage = more exposure
    if data['annual_mileage'] > 20000:
        risk_score += 3
    elif data['annual_mileage'] > 15000:
        risk_score += 1
    
    # Insurance lapses are red flags
    risk_score += data['prior_insurance_lapses'] * 6
    
    # Location risk
    risk_score += data['location_risk_score'] * 10
    
    # Marital status (married = lower risk, statistically)
    if data['marital_status'] == 'Single' and data['age'] < 30:
        risk_score += 2
    
    # Determine denial (threshold around 30-35 points)
    denied = risk_score > 32 or (data['num_claims_3yr'] >= 4) or (data['at_fault_claims'] >= 3)
    
    # Risk levels
    if risk_score < 10:
        risk_level = 'LOW'
        premium_mult = round(random.uniform(0.8, 1.1), 2)
    elif risk_score < 20:
        risk_level = 'MEDIUM'
        premium_mult = round(random.uniform(1.1, 1.5), 2)
    elif risk_score < 30:
        risk_level = 'HIGH'
        premium_mult = round(random.uniform(1.5, 2.2), 2)
    else:
        risk_level = 'VERY_HIGH'
        premium_mult = round(random.uniform(2.2, 3.0), 2)
    
    return denied, risk_level, premium_mult

def generate_correlated_record():
    """Generate a single realistic insurance application record"""
    
    # Age distribution (normal-ish around 35-45)
    age = int(random.gauss(40, 15))
    age = max(16, min(85, age))
    
    # Years licensed (can't exceed driving age)
    max_years = age - 16
    years_licensed = min(max_years, int(random.gauss(max_years * 0.6, 8)))
    years_licensed = max(0, years_licensed)
    
    # Gender distribution
    gender = random.choice(['Male', 'Female', 'Male', 'Female', 'Other'])
    
    # Marital status (age-correlated)
    if age < 25:
        marital_status = random.choice(['Single'] * 8 + ['Married'] * 2)
    elif age < 35:
        marital_status = random.choice(['Single'] * 4 + ['Married'] * 6)
    else:
        marital_status = random.choice(['Married'] * 7 + ['Single'] * 2 + ['Divorced'])
    
    # Claims (most people have 0-1, some have more)
    num_claims_3yr = random.choices([0, 1, 2, 3, 4, 5], 
                                     weights=[50, 25, 12, 7, 4, 2])[0]
    
    # At-fault claims (subset of total claims)
    at_fault_claims = min(num_claims_3yr, 
                          random.choices([0, 1, 2, 3], 
                                       weights=[60, 25, 10, 5])[0])
    
    # Claim amounts (correlated with number of claims)
    if num_claims_3yr == 0:
        total_claim_amount = 0
    else:
        avg_claim = random.uniform(3000, 25000)
        total_claim_amount = round(avg_claim * num_claims_3yr, 2)
    
    # Vehicle age (0-20 years, skewed towards newer)
    vehicle_age = int(random.weibullvariate(5, 1.5))
    vehicle_age = min(20, vehicle_age)
    
    # Annual mileage (normal distribution)
    annual_mileage = int(random.gauss(12000, 4000))
    annual_mileage = max(1000, min(40000, annual_mileage))
    
    # License type (Japan-specific, correlated with experience and claims)
    if years_licensed < 1:
        license_type = 'green'
    elif num_claims_3yr >= 2 or at_fault_claims >= 1:
        license_type = 'blue'
    elif years_licensed >= 5 and num_claims_3yr == 0:
        license_type = 'gold'
    else:
        license_type = random.choice(['blue', 'gold'])
    
    # Insurance lapses (most people have 0)
    prior_insurance_lapses = random.choices([0, 1, 2, 3], 
                                           weights=[75, 15, 7, 3])[0]
    
    # Location risk (0.0 to 1.0, most areas are medium risk)
    location_risk_score = round(random.betavariate(2, 2), 2)
    
    record = {
        'age': age,
        'gender': gender,
        'years_licensed': years_licensed,
        'num_claims_3yr': num_claims_3yr,
        'total_claim_amount': total_claim_amount,
        'at_fault_claims': at_fault_claims,
        'vehicle_age': vehicle_age,
        'annual_mileage': annual_mileage,
        'license_type': license_type,
        'marital_status': marital_status,
        'prior_insurance_lapses': prior_insurance_lapses,
        'location_risk_score': location_risk_score
    }
    
    # Calculate risk profile
    denied, risk_level, premium_mult = calculate_risk_profile(record)
    record['application_denied'] = denied
    record['risk_level'] = risk_level
    record['premium_multiplier'] = premium_mult
    
    return record

def insert_records(num_records=1000):
    """Generate and insert records into database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print(f"🚀 Generating {num_records} insurance records...")
        
        records_inserted = 0
        denied_count = 0
        
        for i in range(num_records):
            record = generate_correlated_record()
            
            cur.execute("""
                INSERT INTO training_data (
                    age, gender, years_licensed, num_claims_3yr,
                    total_claim_amount, at_fault_claims, vehicle_age,
                    annual_mileage, license_type, marital_status,
                    prior_insurance_lapses, location_risk_score,
                    application_denied, risk_level, premium_multiplier
                ) VALUES (
                    %(age)s, %(gender)s, %(years_licensed)s, %(num_claims_3yr)s,
                    %(total_claim_amount)s, %(at_fault_claims)s, %(vehicle_age)s,
                    %(annual_mileage)s, %(license_type)s, %(marital_status)s,
                    %(prior_insurance_lapses)s, %(location_risk_score)s,
                    %(application_denied)s, %(risk_level)s, %(premium_multiplier)s
                )
            """, record)
            
            records_inserted += 1
            if record['application_denied']:
                denied_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  ✓ Inserted {i + 1} records...")
        
        conn.commit()
        
        print(f"\n✅ Successfully inserted {records_inserted} records!")
        print(f"   📊 Denial rate: {denied_count}/{records_inserted} ({denied_count/records_inserted*100:.1f}%)")
        
        # Show distribution
        cur.execute("""
            SELECT risk_level, COUNT(*) 
            FROM training_data 
            GROUP BY risk_level 
            ORDER BY risk_level
        """)
        
        print("\n📈 Risk Level Distribution:")
        for row in cur.fetchall():
            print(f"   {row[0]:12s}: {row[1]} records")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 If you see a connection error, the script needs to run with sudo:")
        print("   sudo -E python3 backend/generate_data.py")

if __name__ == "__main__":
    print("=" * 60)
    print("  Insurance Training Data Generator")
    print("=" * 60)
    insert_records(1000)