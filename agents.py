import json
import urllib.request
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_community.embeddings import FakeEmbeddings
from langchain_chroma import Chroma

from prompts import ROUTER_PROMPT, CATEGORIZER_PROMPT, ADVISOR_PROMPT, LOOKUP_PROMPT
from tools import lookup_transactions

def detect_local_ollama_model() -> str:
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        data = json.loads(req.read().decode())
        models = [m["name"] for m in data.get("models", [])]
        if models:
            return models[0]
    except Exception:
        pass
    return "qwen2.5:1.5b"

@st.cache_resource
def get_llm():
    return ChatOllama(
        model=detect_local_ollama_model(),
        temperature=0.1,
        num_predict=320
    )

@st.cache_resource
def get_retriever():
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma(
        collection_name="budgeting_rules_v1",
        embedding_function=embeddings,
        persist_directory="./chroma_db_store"
    )
    if vectorstore._collection.count() == 0:
        with open("knowledge_base/budgeting_rules.txt", "r", encoding="utf-8") as f:
            rules = [r.strip() for r in f.read().split("\n\n") if r.strip()]
        vectorstore.add_texts(rules)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def _rule_based_category(desc: str) -> str:
    d = desc.upper()
    if any(k in d for k in ["SALARY", "NEFT", "CREDIT", "REFUND"]):
        return "Income"
    if any(k in d for k in ["ZOMATO", "SWIGGY", "CAFE", "RESTAURANT", "FOOD"]):
        return "Food"
    if any(k in d for k in ["UBER", "OLA", "BMTC", "METRO", "IRCTC", "PETROL"]):
        return "Transport"
    if any(k in d for k in ["RENT", "BESCOM", "ELECTRICITY", "JIO", "AIRTEL", "BROADBAND", "BILL"]):
        return "Bills & Utilities"
    if any(k in d for k in ["AMAZON", "MYNTRA", "FLIPKART", "SHOP"]):
        return "Shopping"
    if any(k in d for k in ["NETFLIX", "SPOTIFY", "PVR", "BOOKMYSHOW", "PRIME"]):
        return "Entertainment"
    return "Other"

# AGENT 2: CATEGORIZER AGENT (Batch deduplicated LCEL chain)
def categorize_dataframe(df):
    unique_descriptions = df["Description"].unique().tolist()
    try:
        llm = get_llm()
        categorizer_chain = CATEGORIZER_PROMPT | llm | JsonOutputParser()
        category_map = categorizer_chain.invoke({
            "descriptions_json": json.dumps(unique_descriptions)
        })
        df["Category"] = df["Description"].map(category_map).fillna("Other")
    except Exception:
        df["Category"] = df["Description"].apply(_rule_based_category)
    return df

# AGENT 4: ADVISOR AGENT (Agentic RAG over ChromaDB)
def generate_budget_advice(summary: dict, query: str = "Analyze my spending and flag overspending.") -> str:
    try:
        llm = get_llm()
        retriever = get_retriever()
        docs = retriever.invoke("50/30/20 budget rule food delivery shopping savings")
        context = "\n\n".join([d.page_content for d in docs])

        advisor_chain = ADVISOR_PROMPT | llm | StrOutputParser()
        return advisor_chain.invoke({
            "context": context,
            "query": query,
            "summary": json.dumps(summary, indent=2)
        })
    except Exception:
        savings_pct = round((summary["net_savings"] / summary["total_income"]) * 100, 1) if summary["total_income"] > 0 else 0.0
        return (
            f"Based on the **50/30/20 Budget Rule**, your current savings rate is **{savings_pct}%** "
            f"(₹{summary['net_savings']:,.2f} saved out of ₹{summary['total_income']:,.2f} income)."
        )

# INTENT ROUTER & DISPATCHER (LangChain RunnableBranch)
def handle_user_query(df, summary: dict, user_query: str) -> tuple[str, str]:
    q_lower = user_query.lower()
    if any(w in q_lower for w in ["zomato", "swiggy", "uber", "ola", "netflix", "spotify", "amazon", "myntra", "rent", "show", "find", "list", "orders"]):
        intent = "LOOKUP"
    elif any(w in q_lower for w in ["total", "how much", "breakdown", "sum", "spent", "summary"]):
        intent = "ANALYSIS"
    else:
        try:
            llm = get_llm()
            intent = (ROUTER_PROMPT | llm | StrOutputParser()).invoke({"query": user_query}).strip().upper()
        except Exception:
            intent = "ADVICE"

    def handle_lookup(inputs):
        known_terms = ["Zomato", "Swiggy", "Uber", "Ola", "BMTC", "Netflix", "Spotify", "PVR", "Amazon", "Myntra", "Rent", "BESCOM", "Jio", "Food", "Transport", "Shopping", "Entertainment", "Bills"]
        keyword = next((term for term in known_terms if term.lower() in inputs["query"].lower()), "")

        if not keyword:
            try:
                llm = get_llm()
                keyword = (LOOKUP_PROMPT | llm | StrOutputParser()).invoke({"query": inputs["query"]}).strip().strip('"\'.')
            except Exception:
                stop_words = {"show", "find", "list", "my", "all", "the", "orders", "transactions", "expenses", "for", "me"}
                words = [w for w in inputs["query"].replace("?", "").split() if w.lower() not in stop_words]
                keyword = words[0] if words else ""

        matches = lookup_transactions(df, keyword)
        if matches.empty:
            return f"No transactions matched **`{keyword}`** in this statement."

        total_matched = matches["Amount"].sum()
        return (
            f"Found **{len(matches)}** transaction(s) for **`{keyword}`** — Total: **₹{total_matched:,.2f}**\n\n"
            + matches.to_markdown(index=False)
        )

    def handle_analysis(inputs):
        lines = [
            f"- **Total Income:** ₹{summary['total_income']:,.2f}",
            f"- **Total Expenses:** ₹{summary['total_spent']:,.2f}",
            f"- **Net Savings:** ₹{summary['net_savings']:,.2f}\n",
            "**Exact Category Breakdown:**"
        ]
        for cat, amt in summary["category_totals"].items():
            pct = summary["category_percentages"][cat]
            lines.append(f"- **{cat}:** ₹{amt:,.2f} ({pct}% of income)")
        return "\n".join(lines)

    def handle_advice(inputs):
        return generate_budget_advice(summary, inputs["query"])

    branch = RunnableBranch(
        (lambda x: "LOOKUP" in x["intent"], handle_lookup),
        (lambda x: "ANALYSIS" in x["intent"], handle_analysis),
        handle_advice
    )

    if "LOOKUP" in intent:
        badge = "LOOKUP · DataFrame Filter"
    elif "ANALYSIS" in intent:
        badge = "ANALYSIS · Pandas Math Engine"
    else:
        badge = "ADVICE · ChromaDB RAG Coach"

    return badge, branch.invoke({"intent": intent, "query": user_query})