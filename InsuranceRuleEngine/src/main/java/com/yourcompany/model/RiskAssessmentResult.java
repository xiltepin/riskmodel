package com.yourcompany.model;

public class RiskAssessmentResult {
    private double locationRiskScore;
    private double basePremium;
    private double finalPremium;

    // Constructor
    public RiskAssessmentResult() {
    }

    // Getters and Setters
    public double getLocationRiskScore() {
        return locationRiskScore;
    }

    public void setLocationRiskScore(double locationRiskScore) {
        this.locationRiskScore = locationRiskScore;
    }

    public double getBasePremium() {
        return basePremium;
    }

    public void setBasePremium(double basePremium) {
        this.basePremium = basePremium;
    }

    public double getFinalPremium() {
        return finalPremium;
    }

    public void setFinalPremium(double finalPremium) {
        this.finalPremium = finalPremium;
    }

    @Override
    public String toString() {
        return "RiskAssessmentResult{" +
                "locationRiskScore=" + locationRiskScore +
                ", basePremium=" + basePremium +
                ", finalPremium=" + finalPremium +
                '}';
    }
}