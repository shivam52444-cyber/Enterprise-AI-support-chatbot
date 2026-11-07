# Production-Grade Zero-Trust E-Commerce AI Agent

An enterprise-ready, highly secured autonomous AI Customer Support Agent built for absolute data isolation and compliance at scale. Moving away from basic LLM wrappers, this system introduces a strict **Zero-Trust Architecture** coupled with automated deterministic evaluation pipelines (`openai/gpt-oss-120b`) to securely handle complex transactional workflows.

---

## 🛡️ Core Security Architecture & Safeguard Engine

### 1. Hardened DB Security (Total Data Isolation)
*   **Zero Raw Table Access:** The LLM is completely isolated from raw tables and is strictly restricted to low-privilege database views.
*   **Mutation Protection:** Mitigates risk of prompt-injected `DELETE`, `UPDATE`, or malicious data mutations.

### 2. Ironclad Session Authentication
*   **JWT Token Decoding:** Completely ignores user-supplied text IDs within the chatbox to block account spoofing and data leaks.
*   **Dynamic Context Mapping:** Seamlessly decodes the frontend JWT session token natively on the backend to accurately resolve and map the verified Customer ID.

### 3. Split-Execution Financial Engine & Anti-Fraud Ledger
*   **Autonomous Refunds:** Confidently processes instant, real-time refunds completely autonomously for lower-risk items capped up to **$30**.
*   **HITL Escalation Key:** Claims exceeding **$30** trigger an instant state freeze, auto-generate a unique tracking key, and dynamically route the action to Human-In-The-Loop (HITL) logs.
*   **Immutable Ledger:** Every single financial transaction writes a unique constraint to an isolated ledger database table, preventing duplicate refund exploits ("double-dipping").

---

## 📊 Automated Evaluation & Metrics Breakdown

The agent's safety policies, RAG retrieval accuracy, and guardrails were benchmarked using an automated LLM-as-a-judge pipeline (`openai/gpt-oss-120b`) across 3 explicit test suites (Baseline, Guardrails/Smoke, and Edge Policies) mapped over a real-world dataset of **100,000+ orders (1.5M+ complex relational rows)**.

### 📈 Aggregate Performance (Valid Execution Window)
*Note: Cases p13–p19 hit upstream API rate limits during testing and were excluded from this batch aggregation.*

| Evaluation Metric | Aggregate Score | System Behavior Insights |
| :--- | :---: | :--- |
| **Faithfulness (Anti-Hallucination)** | **98.18%** | **Near Perfect.** The agent strictly adhered to the retrieved text bounds, maintaining near-zero hallucination. |
| **Answer Relevancy** | **93.02%** | **Highly Actionable.** Support responses directly matched customer query intents with negligible verbose filler. |
| **Contextual Precision** | **87.80%** | **Strong Vector Retrieval.** `pgvector` accurately ranked and matched localized enterprise KB documents. |
| **Contextual Recall** | **84.85%** | **High Structural Extraction.** Successfully recovered critical policy rules across both English and Hinglish inputs. |
| **Semantic Correctness** | **67.50%** | **Identified Gaps.** Highlighted specific edge cases where the LLM skipped complex conditional clauses (e.g., product packaging requirements during change-of-mind claims). |

### 🛠️ Production-Ready Guardrails & Validation (Smoke Test)
*   **Intent Isolation:** Passed with a **1.00 (100%) match** on the guardrail suite, successfully isolating generic greeting sequences ("hi", "good morning") from triggering multi-agent business policies.
*   **Multilingual Capability:** Proven robustness in understanding and processing mixed-lingual inputs (such as Roman-script Hinglish queries like *"Mere debit card ka refund kitne din me aayega?"*) while fetching correct policy windows.

---

## ⚡ Production-Ready Tech Stack

*   **Backend Framework:** FastAPI (Asynchronous, blazing-fast concurrent query orchestration)
*   **Database Infrastructure:** AWS PostgreSQL (Heavy-duty production transactional layer)
*   **Vector Search Engine:** `pgvector` (High-performance semantic vector lookup for real-time enterprise policies)
*   **Embedding Model:** `BAAI/bge-small-en-v1.5`
*   **LLM & Judge Pipeline:** `openai/gpt-oss-120b`

*   **Datasetlink**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
*   **querydataset**: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
