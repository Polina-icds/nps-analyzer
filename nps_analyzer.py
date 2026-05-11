import pandas as pd
from transformers import pipeline

# Load sentiment analysis model (Russian)
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
except Exception as e:
    print("Model loading error:", e)
    exit()

# Read feedback CSV
try:
    df = pd.read_csv("feedbacks.csv")
except FileNotFoundError:
    print("feedbacks.csv not found. Please make sure it's in the same folder.")
    exit()
except Exception as e:
    print("CSV read error:", e)
    exit()

# Validate column
if "feedback" not in df.columns:
    print("CSV must contain a 'feedback' column.")
    exit()

# Clean data: convert to string, drop empty
df["feedback"] = df["feedback"].astype(str)
df["feedback"] = df["feedback"].replace("nan", "")
df = df[df["feedback"].str.len() > 5]

if df.empty:
    print("No valid feedback found after cleaning.")
    exit()

# Helper function: from sentiment score to NPS category
def score_to_nps(score):
    if score > 0.5:
        return "Promoter"
    elif score < -0.5:
        return "Detractor"
    else:
        return "Neutral"

# Analyze each feedback
results = []
for _, row in df.iterrows():
    feedback = row["feedback"]
    try:
        result = sentiment_pipeline(feedback)[0]
        label = result["label"]
        confidence = result["score"]
        signed_confidence = confidence if label == "POSITIVE" else -confidence
        nps_cat = score_to_nps(signed_confidence)

        results.append({
            "feedback": feedback,
            "sentiment": label,
            "confidence": confidence,
            "nps_category": nps_cat
        })
    except Exception:
        # silently skip errors during analysis
        pass

if not results:
    print("No feedback could be analyzed.")
    exit()

# Display results
results_df = pd.DataFrame(results)
print("\n🔍 Analysis results:\n")
print(results_df.to_string(index=False))

# Calculate NPS
promoters = results_df[results_df["nps_category"] == "Promoter"].shape[0]
detractors = results_df[results_df["nps_category"] == "Detractor"].shape[0]
total = results_df.shape[0]
nps_score = (promoters - detractors) / total * 100

print(f"\n📊 NPS Score: {nps_score:.1f}")
print(f"👍 Promoters: {promoters}")
print(f"👎 Detractors: {detractors}")
print(f"😐 Neutrals: {total - promoters - detractors}")