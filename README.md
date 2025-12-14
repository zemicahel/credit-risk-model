# Credit Risk Scoring Model

A comprehensive credit scoring model built to meet Basel II regulatory standards, incorporating rigorous data analysis, feature engineering, and model interpretability.

## 📂 Project Structure

```text
credit-risk-model/
├── .github/workflows/   # CI/CD pipelines
├── data/                # Data storage (ignored by git)
│   ├── raw/             # Raw dataset
│   └── processed/       # Cleaned data for modeling
├── notebooks/           # Jupyter Notebooks
│   └── eda.ipynb        # Task 2: Exploratory Data Analysis
├── scripts/             # Modularized helper scripts for EDA
│   ├── loader.py        # Data loading and type conversion
│   └── visualize.py     # Plotting logic for EDA
├── src/                 # Application Source Code (Task 3+)
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
├── tests/               # Unit tests
├── Dockerfile           # Containerization setup
├── docker-compose.yml   # Docker services
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
📊 Task 1: Business Understanding & Regulatory Framework

This section outlines the regulatory framework, business constraints, and modeling decisions required for building a compliant credit risk model under the Basel II Accord.

1. Basel II and the Shift Toward Internal Risk Models

The Basel II Accord marked a fundamental shift in banking regulation by allowing financial institutions to calculate regulatory capital using their own Internal Ratings-Based (IRB) approaches, rather than relying solely on standardized risk weights.

While this provides capital efficiency, it introduces a strict burden of proof. Banks must estimate key risk parameters—Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)—and demonstrate total transparency to regulators. Consequently, interpretability and documentation are compliance requirements, not optional features.

2. Pillar 2 and the "No Black Box" Requirement

Under Pillar 2 (Supervisory Review), regulators generally will not approve "black box" models (e.g., unconstrained neural networks).

2.1 Interpretability as a Regulatory Expectation

Models must exhibit monotonicity and economic intuition.

Example: If a borrower’s debt-to-income ratio increases, the model must demonstrate a corresponding increase in PD. If this cannot be explained, the model fails to align with basic economic reasoning.

2.2 Documentation as an Audit Trail

Banks must provide a "trail of evidence" explaining:

Why variables were selected or excluded.

How variables were transformed.

How the model was estimated and calibrated.

3. The Use Test

The Use Test mandates that models used for regulatory capital must also be actively used in internal risk management and business decision-making.

Implications for Interpretability: Credit officers and senior management must trust the model to use it for loan approvals and limit setting. Opaque models are rarely adopted by stakeholders, causing the model to fail the Use Test.

Implications for Documentation: Operational use requires clear documentation of override policies. If exceptions occur without documented governance, regulators may conclude the model is not genuinely embedded in the business.

4. Independent Model Validation & Stress Testing
4.1 Independent Validation

Basel II requires validation by a team independent of the developers. Documentation acts as the "Instruction Manual" allowing validators to:

Replicate the model.

Stress-test assumptions.

Detect overfitting and spurious correlations.

4.2 Stress Testing (Pillar 2)

To simulate macroeconomic shocks (e.g., a rise in unemployment), banks must understand how specific drivers flow through the model equation. Opaque models make defensible scenario analysis impossible.

5. Pillar 3: Market Discipline

Banks must disclose their risk management practices to investors and rating agencies.

Disclosure: Robust internal documentation is the prerequisite for accurate public disclosure.

Trust: Transparent modeling frameworks signal strong governance, potentially reducing the bank's cost of capital.

6. The Target Variable: Why Create a Proxy?

Real-world banking data rarely contains a "True Risk" label. We must construct a proxy to convert operational data into a binary target (0 = Good, 1 = Bad).

Why a Proxy is Necessary

Binary Transformation: Converting continuous status (e.g., Days Past Due) into a binary classification.

Economic vs. Technical Default: Capturing true distress even if a customer is technically current via minimum payments.

Low-Default Portfolios: In safe portfolios (e.g., mortgages), waiting for final write-offs yields insufficient data. Proxies like "90 Days Past Due" allow for earlier detection.

Business Risks of an Ill-Defined Proxy

If the proxy is flawed, the model optimizes for the wrong objective.

The "Goldilocks" Risk:

Proxy too soft (e.g., 30 DPD): High false positives → Rejecting profitable customers.

Proxy too hard (e.g., Write-off only): High false negatives → Underestimating risk & capital shortfalls.

Learned Bias: If default is defined by internal policy (e.g., "Refer to Collections"), the model learns the policy, not the risk. (e.g., "VIPs never default" because VIPs are never sent to collections).

The Cure Rate Trap: Borrowers who trigger the proxy but return to good standing ("cure") can distort LGD estimates if treated as total losses.

7. Model Selection: Logistic Regression (WoE) vs. Gradient Boosting

In a regulated context, model choice is a trade-off between predictive lift and operational risk.

Performance vs. Explainability

Gradient Boosting (GBM): Higher predictive power (+2–5 Gini); captures non-linearities automatically; opaque structure.

Logistic Regression w/ WoE: Transparent (
𝑦
=
𝑚
𝑥
+
𝑏
y=mx+b
); handles non-linearity via binning; produces standard scorecards; may miss complex interactions.

Regulatory Constraints

Monotonicity: WoE enforces this by design. GBMs require constraints to prevent non-intuitive "jagged" risk surfaces.

Adverse Action: Scorecards easily explain denials to customers (e.g., "High DTI"). GBMs require complex computational attribution (e.g., SHAP) for every decision.

Stability

LR w/ WoE: Coarse binning loses information but gains stability and graceful degradation over time.

GBM: Prone to overfitting noise; can be fragile on Out-of-Time (OOT) data without rigorous regularization.

Deployment

LR w/ WoE: SQL or lookup tables; low latency.

GBM: Requires specialized scoring engines/runtimes; higher IT overhead.

Industry Standard: The Strategic Compromise

Most banks adopt a Hybrid Approach:

Logistic Regression is used for Regulatory Capital and Loan Origination (Compliance & Explainability).

GBM is used for Fraud Detection and Marketing (Performance).

GBMs are often used as "Challenger Models" to benchmark the Logistic Regression.

🔍 Task 2: Exploratory Data Analysis (EDA)

The objective of this phase was to explore the dataset, uncover patterns, and clean the data to prepare it for feature engineering.

Key Activities

Overview & Statistics: Analyzed data structure, types, and central tendencies.

Distribution Analysis: Visualized numerical skewness and categorical imbalances.

Correlation Analysis: Identified relationships between variables (e.g., Amount vs. Value) to prevent multicollinearity.

Missing Values & Outliers: Detected data quality issues requiring imputation or capping.

Modularization

To ensure code quality and reproducibility, the EDA logic was abstracted from the notebook into reusable scripts:

scripts/loader.py: Handles data ingestion and datetime conversion.

scripts/visualize.py: Contains functions for plotting distributions, correlation matrices, and boxplots.

The findings from this analysis will drive the Feature Engineering phase in Task 3.

code
Code
download
content_copy
expand_less
---

### 2. Git Commands to Update and Push

Since you are currently working on `task-2` (EDA) but want the README to reflect the structure of the *entire* project, you should commit this change to the `task-2` branch.

Run the following commands in your terminal:

```bash
# 1. Ensure you are on the task-2 branch
git checkout task-2

# 2. Add the updated README
git add README.md

# 3. Add the scripts and notebooks (if not already added)
git add scripts/ notebooks/

# 4. Commit the changes
git commit -m "Docs: Update README with Business Understanding, EDA details, and Project Structure"

# 5. Push to remote task-2 branch
git push origin task-2
Result on GitHub

Branch task-2: Will show the scripts/, notebooks/, and the fully updated README.md.
