package com.yourcompany.insurance;

import com.yourcompany.model.RiskAssessmentResult;
import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

public class InsuranceApp {
    public static void main(String[] args) {
        try {
            // Load the knowledge base
            KieServices ks = KieServices.Factory.get();
            KieContainer kContainer = ks.getKieClasspathContainer();
            KieSession kSession = kContainer.newKieSession("rulesSession");
            
            // Create a risk assessment
            RiskAssessmentResult result = new RiskAssessmentResult(0.15); // 15% risk
            
            // Insert into session and fire rules
            kSession.insert(result);
            kSession.fireAllRules();
            
            // Display results
            System.out.println("Final Premium: $" + result.getFinalPremium());
            
            // Clean up
            kSession.dispose();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}