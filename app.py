from pathlib import Path
import pandas as pd
import streamlit as st

from tools import parse_statement, compute_financial_summary
from agents import categorize_dataframe, generate_budget_advice, handle_user_query, detect_local_ollama_model

st.set_page_config(
    page_title="FinLedger · Statement & Cashflow Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1180px; }
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric-title { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-num { font-size: 1.7rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem; }
    .metric-tag-good { font-size: 0.78rem; color: #059669; font-weight: 500; margin-top: 0.2rem; }
    .metric-tag-neutral { font-size: 0.78rem; color: #475569; font-weight: 500; margin-top: 0.2rem; }
    .route-pill {
        display: inline-block;
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### FinLedger")
    st.caption("Statement Analytics & Budget Intelligence")
    st.divider()

    uploaded_file = st.file_uploader(
        "Import Bank / UPI Statement",
        type=["csv", "pdf"],
        help="Upload a CSV or PDF statement, or explore the preloaded sample statement."
    )

    if st.button("Load Sample UPI Statement", use_container_width=True):
        st.session_state.pop("active_source", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

    st.divider()
    local_model = detect_local_ollama_model()
    st.markdown("**System Architecture**")
    st.caption(
        f"• **Local LLM:** `{local_model}` (Offline)\n"
        "• **Math Engine:** Deterministic `pandas`\n"
        "• **Rule Store:** Local ChromaDB (`50/30/20`)"
    )

st.markdown("## Cashflow & Statement Overview")
st.caption("Automated merchant tagging, deterministic expense analytics, and rule-grounded financial review.")

sample_path = Path(__file__).parent / "data" / "sample_upi.csv"

try:
    if uploaded_file is not None:
        source_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        raw_transactions = parse_statement(uploaded_file)
        active_label = f"Active Statement: **{uploaded_file.name}**"
    elif sample_path.is_file():
        source_id = "bundled_sample_upi"
        with sample_path.open("rb") as sample_file:
            raw_transactions = parse_statement(sample_file)
        active_label = "Active Statement: **Sample Indian UPI Statement** (import your own CSV/PDF in the sidebar)"
    else:
        st.info("Import a CSV or PDF statement from the sidebar to start.")
        st.stop()

    required_columns = {"Date", "Description", "Amount", "Type"}
    missing_columns = required_columns.difference(raw_transactions.columns)
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}.")
        st.stop()

    if st.session_state.get("active_source") != source_id:
        with st.spinner("Tagging merchants and computing cashflow summary..."):
            categorized_df = categorize_dataframe(raw_transactions.copy())
            summary = compute_financial_summary(categorized_df)
            advice = generate_budget_advice(summary)

            st.session_state["active_source"] = source_id
            st.session_state["transactions"] = categorized_df
            st.session_state["summary"] = summary
            st.session_state["advice"] = advice
            st.session_state["chat_history"] = []

    transactions = st.session_state["transactions"]
    summary = st.session_state["summary"]

    st.caption(active_label)

    savings_rate = round((summary["net_savings"] / summary["total_income"]) * 100, 1) if summary["total_income"] > 0 else 0.0
    expense_rate = round((summary["total_spent"] / summary["total_income"]) * 100, 1) if summary["total_income"] > 0 else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Income</div>
            <div class="metric-num">₹{summary['total_income']:,.2f}</div>
            <div class="metric-tag-good">Verified statement credits</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Outflow</div>
            <div class="metric-num">₹{summary['total_spent']:,.2f}</div>
            <div class="metric-tag-neutral">{expense_rate}% of total income</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Net Savings</div>
            <div class="metric-num">₹{summary['net_savings']:,.2f}</div>
            <div class="metric-tag-good">{savings_rate}% savings rate (Benchmark ≥ 20%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    tab_overview, tab_ledger, tab_assistant = st.tabs([
        "Spending Breakdown",
        "Transaction Ledger",
        "Budget Advisor & Query Console"
    ])

    with tab_overview:
        c_chart, c_table = st.columns([1.25, 1])
        with c_chart:
            st.subheader("Expenses by Category")
            spending = transactions.loc[transactions["Type"] == "Debit"]
            if not spending.empty:
                cat_series = spending.groupby("Category")["Amount"].sum().sort_values(ascending=False)
                st.bar_chart(cat_series, color="#0f766e", height=280)

        with c_table:
            st.subheader("Category Allocation")
            alloc_df = pd.DataFrame([
                {
                    "Category": cat,
                    "Amount": f"₹{amt:,.2f}",
                    "Share of Income": f"{summary['category_percentages'][cat]}%"
                }
                for cat, amt in summary["category_totals"].items()
            ])
            st.dataframe(alloc_df, use_container_width=True, hide_index=True, height=280)

        st.divider()
        st.subheader("Daily Spending Trend")
        peak = summary.get("peak_spend_day", {"date": "N/A", "amount": 0.0})
        st.caption(f"Peak outflow recorded on **{peak['date']}** (₹{peak['amount']:,.2f}).")
        if summary.get("daily_trends"):
            trend_series = pd.Series(summary["daily_trends"], name="Daily Spend (₹)")
            st.area_chart(trend_series, color="#0284c7", height=220)

    with tab_ledger:
        f1, f2, f3 = st.columns([1.6, 1, 1])
        with f1:
            query_filter = st.text_input("Search merchant or narration", placeholder="Type Zomato, Rent, Amazon, Swiggy...")
        with f2:
            cat_options = ["All Categories"] + sorted(transactions["Category"].unique().tolist())
            chosen_cat = st.selectbox("Category", cat_options)
        with f3:
            chosen_type = st.selectbox("Entry Type", ["All", "Debit", "Credit"])

        view_df = transactions.copy()
        if query_filter:
            view_df = view_df[view_df["Description"].str.contains(query_filter, case=False, na=False)]
        if chosen_cat != "All Categories":
            view_df = view_df[view_df["Category"] == chosen_cat]
        if chosen_type != "All":
            view_df = view_df[view_df["Type"] == chosen_type]

        st.dataframe(view_df, use_container_width=True, hide_index=True, height=350)

        st.download_button(
            label="Download Filtered CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="finledger_categorized.csv",
            mime="text/csv"
        )

    with tab_assistant:
        st.subheader("Automated Budget Review")
        with st.container(border=True):
            st.markdown(st.session_state["advice"])

        st.divider()
        st.subheader("Interactive Statement Assistant")

        b1, b2, b3 = st.columns(3)
        selected_quick = None
        if b1.button("Zomato orders", use_container_width=True):
            selected_quick = "Zomato orders"
        if b2.button("Exact category breakdown", use_container_width=True):
            selected_quick = "Give me the exact category breakdown"
        if b3.button("Where am I overspending?", use_container_width=True):
            selected_quick = "Where am I overspending and how do I fix it?"

        user_input = st.chat_input("Ask to look up merchants, verify totals, or get budgeting advice...")
        final_query = selected_quick or user_input

        if final_query:
            with st.spinner("Processing query..."):
                route_badge, answer_md = handle_user_query(transactions, summary, final_query)
                st.session_state["chat_history"].append({
                    "user": final_query,
                    "badge": route_badge,
                    "assistant": answer_md
                })

        for msg in st.session_state.get("chat_history", []):
            with st.chat_message("user"):
                st.write(msg["user"])
            with st.chat_message("assistant"):
                st.markdown(f'<span class="route-pill">{msg["badge"]}</span>', unsafe_allow_html=True)
                st.markdown(msg["assistant"])

except Exception as error:
    st.error(f"Unable to process statement: {error}")