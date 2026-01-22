package com.yourcompany.service;

import com.yourcompany.model.RiskAssessmentResult;
import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.springframework.stereotype.Service;

@Service
public class DroolsService {

    private final KieContainer kieContainer;

    public DroolsService() {
        KieServices kieServices = KieServices.Factory.get();
        this.kieContainer = kieServices.getKieClasspathContainer();
    }

    public void executePremiumCalculationRules(RiskAssessmentResult result) {
        KieSession kieSession = kieContainer.newKieSession("rulesSession");
        if (kieSession == null) {
            throw new IllegalStateException(
                "KieSession 'rulesSession' not found. Check kmodule.xml"
            );
        }

        try {
            kieSession.insert(result);
            kieSession.fireAllRules();
        } finally {
            kieSession.dispose();
        }
    }
}
