import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ChangeDetectorRef } from '@angular/core';  // ← NUEVO IMPORT
import { environment } from '../../environments/environment';

interface PredictionResult {
  decision: string;
  denial_probability: number;
  risk_level: string;
  premium_multiplier: number;
  confidence: number;
  model_version: string;
  timestamp: string;
}

@Component({
  selector: 'app-risk-assessment',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,
    MatCardModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './risk-assessment.html',
  styleUrls: ['./risk-assessment.scss']
})
export class RiskAssessmentComponent {
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  private cdr = inject(ChangeDetectorRef);  // ← NUEVO: ChangeDetectorRef

  assessmentForm: FormGroup;
  result: PredictionResult | null = null;
  loading = false;

  constructor() {
    this.assessmentForm = this.fb.group({
      age: [35, [Validators.required, Validators.min(16), Validators.max(85)]],
      gender: ['female', Validators.required],
      years_licensed: [15, [Validators.required, Validators.min(0)]],
      num_claims_3yr: [0, Validators.min(0)],
      total_claim_amount: [0, Validators.min(0)],
      at_fault_claims: [0, Validators.min(0)],
      vehicle_age: [3, Validators.min(0)],
      annual_mileage: [12000, Validators.min(0)],
      license_type: ['blue', Validators.required],  // Default to standard
      marital_status: ['married'],
      prior_insurance_lapses: [0, Validators.min(0)],
      location_risk_score: [0.3, [Validators.min(0), Validators.max(1)]]
    });
  }

  onSubmit() {
    if (this.assessmentForm.invalid || this.loading) return;

    this.loading = true;
    this.result = null;

    const payload = {
      age: Number(this.assessmentForm.value.age),
      gender: this.assessmentForm.value.gender.toLowerCase(),
      years_licensed: Number(this.assessmentForm.value.years_licensed),
      num_claims_3yr: Number(this.assessmentForm.value.num_claims_3yr || 0),
      total_claim_amount: Number(this.assessmentForm.value.total_claim_amount || 0),
      at_fault_claims: Number(this.assessmentForm.value.at_fault_claims || 0),
      vehicle_age: Number(this.assessmentForm.value.vehicle_age || 5),
      annual_mileage: Number(this.assessmentForm.value.annual_mileage || 12000),
      license_type: this.assessmentForm.value.license_type.toLowerCase(),
      marital_status: this.assessmentForm.value.marital_status.toLowerCase(),
      prior_insurance_lapses: Number(this.assessmentForm.value.prior_insurance_lapses || 0),
      location_risk_score: Number(this.assessmentForm.value.location_risk_score || 0.5)
    };

    this.http.post<PredictionResult>(`${environment.apiUrl}/api/predict`, payload)
      .subscribe({
        next: (res) => {
          console.log('✅ Respuesta recibida:', res);  // Para verificar en consola
          this.result = res;
          this.loading = false;
          this.cdr.detectChanges();  // ← FORZAR DETECCIÓN DE CAMBIOS
        },
        error: (err) => {
          console.error('❌ Error en predicción:', err);
          alert('Connection error. Is the backend running on port 5001?');
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }

  getRiskClass(): string {
    if (!this.result) return '';
    switch (this.result.risk_level) {
      case 'LOW': return 'low-risk';
      case 'MEDIUM': return 'medium-risk';
      case 'HIGH': return 'high-risk';
      case 'VERY_HIGH': return 'very-high-risk';
      default: return '';
    }
  }

  getRiskTextClass(): string {
    return 'risk-text ' + this.getRiskClass();
  }
}