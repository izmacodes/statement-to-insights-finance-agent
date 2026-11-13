import io
import pandas as pd
import pdfplumber

def parse_statement(uploaded_file) -> pd.DataFrame:
    filename = getattr(uploaded_file, "name", "statement.csv").lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".pdf"):
        rows = []
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    rows.extend(table)
        if len(rows) < 2:
            raise ValueError("No readable transaction table found in the PDF.")
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        raise ValueError("Unsupported file format. Please upload CSV or PDF.")

    df.columns = [str(c).strip().title() for c in df.columns]
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Type"] = df["Type"].astype(str).str.strip().str.title()
    return df

def compute_financial_summary(df: pd.DataFrame) -> dict:
    income = float(df.loc[df["Type"] == "Credit", "Amount"].sum())
    expenses_df = df.loc[df["Type"] == "Debit"].copy()
    total_spent = float(expenses_df["Amount"].sum())
    net_savings = income - total_spent

    category_totals = (
        expenses_df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    category_percentages = {
        cat: round((amt / income) * 100, 1) if income > 0 else 0.0
        for cat, amt in category_totals.items()
    }

    daily_trends = (
        expenses_df.groupby("Date")["Amount"]
        .sum()
        .sort_index()
        .to_dict()
    )

    peak_spend_day = max(daily_trends.items(), key=lambda x: x[1]) if daily_trends else ("N/A", 0.0)

    top_expenses = (
        expenses_df.nlargest(3, "Amount")[["Date", "Description", "Category", "Amount"]]
        .to_dict(orient="records")
    )

    return {
        "total_income": income,
        "total_spent": total_spent,
        "net_savings": net_savings,
        "category_totals": category_totals,
        "category_percentages": category_percentages,
        "daily_trends": daily_trends,
        "peak_spend_day": {"date": peak_spend_day[0], "amount": float(peak_spend_day[1])},
        "top_expenses": top_expenses,
    }

def lookup_transactions(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if not keyword:
        return pd.DataFrame()
    mask = (
        df["Description"].astype(str).str.contains(keyword, case=False, na=False)
        | df["Category"].astype(str).str.contains(keyword, case=False, na=False)
    )
    return df.loc[mask]