# NPS Analyzer

A simple Python tool to analyze customer feedback sentiment and calculate NPS (Net Promoter Score) using a pre‑trained sentiment model from Hugging Face.

## How it works

1. Loads a pre‑trained sentiment model: `blanchefort/rubert-base-cased-sentiment`
2. Reads customer feedback from `feedbacks.csv`
3. Detects sentiment: `POSITIVE`, `NEGATIVE`, or `NEUTRAL`
4. Converts sentiment to NPS categories: **Promoter**, **Neutral**, **Detractor**
5. Calculates NPS Score

## Setup & Run

1. Clone the repository and open the folder.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   ## Web Interface

The project also includes a **Streamlit** app for interactive analysis:

```bash
streamlit run app.py