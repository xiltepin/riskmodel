package com.yourcompany;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class InsuranceRulesApplication {

    public static void main(String[] args) {
        SpringApplication.run(InsuranceRulesApplication.class, args);
        System.out.println("🚀 Insurance Rules Engine started on port 5001");
        System.out.println("📋 Drools rules loaded successfully");
    }
}