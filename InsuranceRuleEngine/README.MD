It assumes:

Angular has its own README

Flask (if any) has its own README

This README lives at the root of InsuranceRuleEngine/

You can copy-paste this directly as README.md.

Insurance Rules Engine (Java + Drools)

This module is a Spring Boot–based Insurance Rules Engine that uses Drools to calculate insurance premiums based on risk factors.

It exposes REST endpoints consumed by the Angular frontend and evaluates business rules defined in .drl files.

📦 Tech Stack

Java: 21

Spring Boot: 3.x

Drools: 8.x

Maven: 3.9+

Embedded Tomcat

📁 Project Structure
InsuranceRuleEngine/
├── src/main/java
│   └── com/yourcompany
│       ├── InsuranceRulesApplication.java
│       ├── controller/
│       ├── service/
│       │   └── DroolsService.java
│       └── model/
│           └── RiskAssessmentResult.java
│
├── src/main/resources
│   ├── application.properties
│   ├── META-INF/
│   │   └── kmodule.xml
│   └── com/yourcompany/insurance/
│       └── InsuranceRules.drl
│
├── pom.xml
└── README.md

✅ Prerequisites

Make sure the following are installed:

Java
java -version


Expected:

openjdk 21.x


If needed:

sudo apt install openjdk-21-jdk

Maven
mvn -version


Expected:

Apache Maven 3.9+


If needed:

sudo apt install maven

⚙️ Drools Configuration
kmodule.xml

Location:

src/main/resources/META-INF/kmodule.xml


Example:

<?xml version="1.0" encoding="UTF-8"?>
<kmodule xmlns="http://www.drools.org/xsd/kmodule">
    <kbase name="insuranceKBase" packages="com.yourcompany.insurance">
        <ksession name="rulesSession"/>
    </kbase>
</kmodule>


⚠️ Important
The packages value must match the package declared inside your .drl files.

Rules File (.drl)

Location:

src/main/resources/com/yourcompany/insurance/InsuranceRules.drl


This file contains all Drools business rules.

🧪 Clean & Build the Project

Always clean before running if rules or config changed:

mvn clean package


If successful, you’ll see:

BUILD SUCCESS

▶️ Run the Application (Recommended)
✅ Run as a packaged JAR (PRODUCTION-LIKE)
java -jar target/insurance-rules-engine-1.0-SNAPSHOT.jar


This is recommended because:

Uses the same classpath as production

Correctly loads Drools rules from BOOT-INF/classes

Avoids classloader issues

⚠️ Alternative: Run via Maven (DEV ONLY)
mvn spring-boot:run


Use only for quick development.
Drools behaves more reliably when run as a packaged JAR.

🌐 Application Ports

Spring Boot HTTP: 8080

(Angular connects to this backend)

Example startup log:

🚀 Insurance Rules Engine started on port 8080
📋 Drools rules loaded successfully

🔁 Common Commands
Clean everything
mvn clean

Rebuild
mvn clean package

Run tests (if any)
mvn test

🧠 Drools Rule Execution Flow

Angular sends a request

Spring Controller receives input

DroolsService creates a KieSession

Rules are evaluated

Final premium is calculated

Response is returned to frontend

🚨 Common Issues & Fixes
❌ Infinite rule execution

Cause:

Missing no-loop true

Rules update facts without guard conditions

Fix:

rule "Example"
no-loop true
when
    finalPremium == 0
then
    setFinalPremium(650);
    update($fact);
end

❌ Unknown KieSession name

Cause:

Session name mismatch with kmodule.xml

Fix:
Ensure:

kieContainer.newKieSession("rulesSession");


matches:

<ksession name="rulesSession"/>

❌ Missing MVEL dependency

Error:

MissingDependencyException: add drools-mvel


Fix (pom.xml):

<dependency>
  <groupId>org.drools</groupId>
  <artifactId>drools-mvel</artifactId>
</dependency>

🚫 Git Ignore Recommendation

Add to .gitignore:

target/


Never commit build artifacts.

📌 Notes for Developers

Do not put rules under src/main/java

Rules must live under src/main/resources

Always rebuild after modifying .drl files

Prefer field constraints over eval()