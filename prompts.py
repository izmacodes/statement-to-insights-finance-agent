from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's finance question into EXACTLY ONE word:
- LOOKUP (finding/filtering specific merchants, orders, or category rows)
- ANALYSIS (asking for exact totals, sums, percentages, or spending breakdown)
- ADVICE (asking for budgeting tips, overspending alerts, or how to save money)

Reply with ONLY that single word."""),
    ("user", "{query}")
])

CATEGORIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Map each raw bank/UPI description to ONE exact category from:
[Food, Transport, Bills & Utilities, Shopping, Entertainment, Income, Other]

Few-Shot Examples:
- "NEFT-SALARY-TECHCORP" -> "Income"
- "UPI/428910/ZOMATO/ORDER" -> "Food"
- "UPI/991823/BMTC BUS PASS" -> "Transport"
- "UPI/771234/BESCOM ELECTRICITY BILL" -> "Bills & Utilities"
- "UPI/112938/AMAZON PAY SHOPPING" -> "Shopping"
- "ACH/NETFLIX.COM/AUTOPAY" -> "Entertainment"

Return ONLY a valid JSON object mapping each input description to its category."""),
    ("user", "Categorize these descriptions:\n{descriptions_json}")
])

ADVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a practical Personal Finance Advisor.
Using the Budgeting Rules from ChromaDB and the exact Pandas Financial Summary below:
1. Compare the user's spending percentages against the 50/30/20 and category rules.
2. Flag any overspending risks clearly.
3. Give 3 specific, actionable rupee-saving recommendations tailored to their exact numbers.
Never recalculate or invent numbers—use only the provided Pandas figures.

Budgeting Rules:
{context}"""),
    ("user", "Question: {query}\n\nExact Financial Summary:\n{summary}")
])

LOOKUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract the single search keyword (merchant like Zomato, Swiggy, Uber, Netflix, Rent, Amazon, Myntra, OR category like Food, Transport, Shopping, Bills, Entertainment) from the query.
Return ONLY that single word."""),
    ("user", "{query}")
])