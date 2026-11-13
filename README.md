# Statement-to-Insights Finance Agent (`FinLedger`)

An offline-first personal finance assistant that ingests bank and UPI statements (CSV or PDF), categorizes raw transaction narrations using a local LLM, computes exact cashflow metrics and daily spending trends with `pandas`, and delivers rule-grounded budgeting insights via ChromaDB Agentic RAG and LangChain LCEL intent routing.

---

## Architecture & Agent Design

The system uses a **neuro-symbolic architecture**—delegating semantic classification and natural-language coaching to a local LLM while keeping all arithmetic strictly deterministic in Python (`pandas`) to prevent numerical hallucinations.

| Component | Responsibility | Implementation |
| :--- | :--- | :--- |
| **Parser Agent** | Ingests raw CSV and PDF statements, extracts transaction tables, and normalizes `Date`, `Description`, `Amount`, and `Type` fields. | `tools.py` (`parse_statement` via `pandas` & `pdfplumber`) |
| **Categorizer Agent** | Deduplicates unique merchant/UPI strings and classifies them into 7 standardized categories (`Food`, `Transport`, `Bills & Utilities`, `Shopping`, `Entertainment`, `Income`, `Other`) with deterministic rule fallback. | `agents.py` (`categorize_dataframe`) & `prompts.py` (`CATEGORIZER_PROMPT`) |
| **Analysis Agent** | Computes total income, total outflow, net savings, savings rate, category percentages, peak spend day, and daily spending trends. | `tools.py` (`compute_financial_summary` & `lookup_transactions`) |
| **Advisor Agent** | Retrieves personal finance benchmarks (50/30/20 rule, food delivery caps, subscription audits, impulse-buy thresholds) from ChromaDB and generates actionable rupee-saving advice. | `agents.py` (`generate_budget_advice`) & `prompts.py` (`ADVISOR_PROMPT`) |
| **Intent Router** | Classifies user queries and routes execution across `LOOKUP` (DataFrame filter), `ANALYSIS` (`pandas` breakdown), and `ADVICE` (ChromaDB RAG). | `agents.py` (`handle_user_query` via `RunnableBranch`) |

---

## Tech Stack

- **Frontend:** Streamlit (`app.py`)
- **Backend:** Python 3.9+
- **Local LLM Engine:** Ollama (`qwen2.5:1.5b` via `langchain-ollama`) — zero cloud API keys or rate limits
- **Orchestration & Prompting:** LangChain Core (`ChatPromptTemplate`, LCEL `|` pipelines, `JsonOutputParser`, `StrOutputParser`, `RunnableBranch`)
- **Vector Store (RAG):** ChromaDB (`langchain-chroma` persisted locally in `./chroma_db_store`)
- **Data & Math Processing:** `pandas` & `pdfplumber`

---

## Project Structure

```text
statement-to-insights-finance-agent/
├── data/
│   └── sample_upi.csv              # Preloaded 18-transaction Indian UPI sample statement
├── knowledge_base/
│   └── budgeting_rules.txt         # Domain rules (50/30/20 framework, dining & shopping caps)
├── app.py                          # Streamlit dashboard (KPIs, charts, ledger filter, query console)
├── agents.py                       # Local Ollama LLM setup, ChromaDB RAG, LCEL chains & intent router
├── tools.py                        # CSV/PDF statement parser, pandas financial math & lookup tools
├── prompts.py                      # Structured Zero-Shot, Few-Shot, CoT & keyword extraction prompts
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes venv, local ChromaDB store, and cache artifacts
└── README.md                       # Project documentation
