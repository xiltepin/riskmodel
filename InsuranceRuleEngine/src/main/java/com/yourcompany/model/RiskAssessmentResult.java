package com.yourcompany.model;

public class RiskAssessmentResult {
    private double locationRiskScore;
    private double finalPremium;
    
    public RiskAssessmentResult() {}
    
    public RiskAssessmentResult(double locationRiskScore) {
        this.locationRiskScore = locationRiskScore;
    }
    
    public double getLocationRiskScore() {
        return locationRiskScore;
    }
    
    public void setLocationRiskScore(double locationRiskScore) {
        this.locationRiskScore = locationRiskScore;
    }
    
    public double getFinalPremium() {
        return finalPremium;
    }
    
    public void setFinalPremium(double finalPremium) {
        this.finalPremium = finalPremium;
    }
}