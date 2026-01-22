import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { HttpClientModule, HttpClient } from '@angular/common/http';
import { RouterLink } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ChangeDetectorRef } from '@angular/core';
import { environment } from '../../environments/environment';

interface PremiumCalculationResult {
  basePremium: number;
  locationRiskScore: number;
  finalPremium: number;
  riskMultiplier: number;
  executionTime?: number;
}

@Component({
  selector: 'app-rules-engine',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule,
    RouterLink,
    MatCardModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './rules-engine.html',
  styleUrls: ['./rules-engine.scss']
})
export class RulesEngineComponent {
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);
  private cdr = inject(ChangeDetectorRef);

  premiumForm: FormGroup;
  result: PremiumCalculationResult | null = null;
  loading = false;

  constructor() {
    this.premiumForm = this.fb.group({
      locationRiskScore: [0.3, [Validators.required, Validators.min(0), Validators.max(1)]],
      basePremium: [500, [Validators.required, Validators.min(100)]]
    });
  }

  onSubmit() {
    if (this.premiumForm.invalid || this.loading) return;

    this.loading = true;
    this.result = null;

    const payload = {
      locationRiskScore: Number(this.premiumForm.value.locationRiskScore),
      basePremium: Number(this.premiumForm.value.basePremium)
    };

    // Use rulesApiUrl if available, otherwise fall back to apiUrl with different port
    const apiUrl = (environment as any).rulesApiUrl || 'http://localhost:8080';
    
    this.http.post<PremiumCalculationResult>(`${apiUrl}/api/rules/calculate-premium`, payload)
      .subscribe({
        next: (res) => {
          console.log('✅ Premium calculation result:', res);
          this.result = res;
          this.loading = false;
          this.cdr.detectChanges();
        },
        error: (err) => {
          console.error('❌ Error in premium calculation:', err);
          alert('Connection error. Is the Drools backend running?');
          this.loading = false;
          this.cdr.detectChanges();
        }
      });
  }
}