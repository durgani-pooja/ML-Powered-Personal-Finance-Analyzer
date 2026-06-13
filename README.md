# 💰 ML-Powered Personal Finance Analyzer

Upload your bank statement CSV and instantly get:
- 🏷️ Auto-categorized transactions (zero-shot NLP)
- ⚠️ Anomaly detection on unusual spends
- 🔮 30-day expense forecast (Prophet)
- 🤖 Personalized saving tips (LLaMA-3 via Groq)

## 🚀 Live Demo
[Add HuggingFace Spaces link here]

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| NLP Categorization | facebook/bart-large-mnli (zero-shot) |
| Anomaly Detection | Scikit-learn Isolation Forest |
| Forecasting | Facebook Prophet |
| LLM Advice | LLaMA-3 8B via Groq API |
| UI | Streamlit + Plotly |

## 📦 Setup

```bash
git clone https://github.com/YOUR_USERNAME/finance-analyzer
cd finance-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
