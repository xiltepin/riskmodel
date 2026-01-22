package com.yourcompany.controller;

import com.yourcompany.model.RiskAssessmentResult;
import com.yourcompany.service.DroolsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/rules")
@CrossOrigin(origins = "http://localhost:4300")
public class RulesEngineController {

    @Autowired
    private DroolsService droolsService;

    @PostMapping("/calculate-premium")
    public ResponseEntity<Map<String, Object>> calculatePremium(@RequestBody Map<String, Object> request) {
        try {
            double locationRiskScore = ((Number) request.get("locationRiskScore")).doubleValue();
            double basePremium = ((Number) request.get("basePremium")).doubleValue();

            // Create result object
            RiskAssessmentResult result = new RiskAssessmentResult();
            result.setLocationRiskScore(locationRiskScore);
            result.setBasePremium(basePremium);

            // Execute Drools rules
            long startTime = System.currentTimeMillis();
            droolsService.executePremiumCalculationRules(result);
            long executionTime = System.currentTimeMillis() - startTime;

            // Prepare response
            Map<String, Object> response = new HashMap<>();
            response.put("basePremium", basePremium);
            response.put("locationRiskScore", locationRiskScore);
            response.put("finalPremium", result.getFinalPremium());
            response.put("riskMultiplier", 1 + locationRiskScore);
            response.put("executionTime", executionTime);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            e.printStackTrace();
            Map<String, Object> error = new HashMap<>();
            error.put("error", "Failed to calculate premium: " + e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}