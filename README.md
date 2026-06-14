# 💰 ML-Powered Personal Finance Analyzer

A smart personal finance dashboard that turns raw bank statement data into actionable insights. Upload a CSV of your transactions, and the app automatically categorizes your spending, flags unusual transactions, forecasts your future expenses, and gives you AI-powered tips to save money — all through an interactive Streamlit interface.

Upload your bank statement CSV and instantly get:
- 🏷️ Auto-categorized transactions (zero-shot NLP)
- ⚠️ Anomaly detection on unusual spends
- 🔮 30-day expense forecast (Prophet)
- 🤖 Personalized saving tips (LLaMA-3 via Groq)

## 🚀 Live Demo
🔗 **Try it here:** [ml-powered-personal-finance-analyzer.streamlit.app](https://ml-powered-personal-finance-analyzer-fxelyxvbvfi4fxjczy5bqy.streamlit.app/)

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| NLP Categorization | facebook/bart-large-mnli (zero-shot) |
| Anomaly Detection | Scikit-learn Isolation Forest |
| Forecasting | Facebook Prophet |
| LLM Advice | LLaMA-3 8B via Groq API |
| UI | Streamlit + Plotly |

## 📦 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/finance-analyzer
cd finance-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Add your Groq API key to `.env`:
