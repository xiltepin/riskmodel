import { Routes } from '@angular/router';
import { RiskAssessmentComponent } from './risk-assessment/risk-assessment';
import { LandingPageComponent } from './landing-page/landing-page';
import { RulesEngineComponent } from './rules-engine/rules-engine';

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
    path: 'RulesEngine',
    component: RulesEngineComponent
  },
  {
    path: '**',
    redirectTo: ''
  }
];