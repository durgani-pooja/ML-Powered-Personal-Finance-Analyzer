# advisor.py
# Sends your spending summary to Groq (LLaMA-3)
# and gets back realistic, personalised saving tips

from groq import Groq
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_spending_summary(df: pd.DataFrame) -> str:
    """Build a plain-English summary of spending by category."""
    expenses = df[df["Amount"] > 0]
    summary = expenses.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    lines = ["Here is my monthly spending breakdown (in Indian Rupees):"]
    for category, amount in summary.items():
        lines.append(f"- {category}: ₹{amount:,.0f}")

    total = expenses["Amount"].sum()
    lines.append(f"\nTotal monthly spending: ₹{total:,.0f}")

    return "\n".join(lines)

def get_saving_advice(df: pd.DataFrame) -> str:
    """
    Send spending summary to Groq LLaMA-3,
    get back 5 realistic saving tips.
    """
    summary = build_spending_summary(df)

    prompt = f"""
You are a friendly and practical personal finance advisor for a young professional in India.

{summary}

Based on this spending pattern, give me exactly 5 specific, actionable saving tips.
Be realistic, friendly, and mention actual rupee amounts where possible.
Don't be preachy — just helpful and direct.
Format each tip as a numbered point.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600
    )

    return response.choices[0].message.content