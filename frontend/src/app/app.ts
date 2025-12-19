import { Component } from '@angular/core';
import { RiskAssessmentComponent } from './risk-assessment/risk-assessment';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RiskAssessmentComponent
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.scss']
})
export class AppComponent {  // ← Aquí el cambio: AppComponent
  title = 'Insurance Risk Assessment POC';
}