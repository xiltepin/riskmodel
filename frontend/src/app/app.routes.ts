import { Routes } from '@angular/router';
import { RiskAssessmentComponent } from './risk-assessment/risk-assessment';
import { LandingPageComponent } from './landing-page/landing-page';

export const routes: Routes = [
  {
    path: '',
    component: LandingPageComponent
  },
  {
    path: 'RiskAssessment',
    component: RiskAssessmentComponent
  },
  {
    path: '**',
    redirectTo: ''
  }
];