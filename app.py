from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

df = pd.read_excel("survey_csp_veerpuram.xlsx")
model = joblib.load("garbage_model.pkl")

GOOGLE_MAP_EMBED = "https://www.google.com/maps/d/embed?mid=1eL6-Cikse4JZuEl_glOh45W3Ki6nqhU&ehbc=2E312F"

def predict_risk(avg_kg):
    inp = pd.DataFrame([[avg_kg]], columns=["Garbage Produced (kg)/Month"])
    prob = float(model.predict_proba(inp)[0][1])
    level = ("CRITICAL" if prob >= 0.80 else
             "HIGH"     if prob >= 0.50 else
             "MODERATE" if prob >= 0.25 else "LOW")
    rec = (
        "IMMEDIATE MUNICIPAL INTERVENTION REQUIRED. Deploy additional waste collection units and enforce segregation."
        if prob >= 0.80 else
        "Schedule urgent waste collection. Alert local Panchayat authorities."
        if prob >= 0.50 else
        "Monitor zone weekly. Run waste awareness drives among residents."
        if prob >= 0.25 else
        "Zone is under control. Maintain current collection schedule."
    )
    return round(prob * 100, 2), level, rec

@app.route("/api/map-url")
def get_map_url():
    return jsonify({"url": GOOGLE_MAP_EMBED})

@app.route("/api/survey")
def get_survey():
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/risk")
def get_risk():
    results = []
    for ap, group in df.groupby("Accumulation Point"):
        households = []
        for _, row in group.iterrows():
            households.append({
                "House Owner": row["House Owner"],
                "Street": row["Street"],
                "Garbage Produced (kg)/Month": float(row["Garbage Produced (kg)/Month"]),
                "Total Members": int(row["Total Members"]),
            })
        avg_kg = float(group["Garbage Produced (kg)/Month"].mean())
        total_kg = float(group["Garbage Produced (kg)/Month"].sum())
        prob, level, rec = predict_risk(avg_kg)
        results.append({
            "accumulation_point": int(ap),
            "total_garbage": round(total_kg, 2),
            "avg_garbage": round(avg_kg, 2),
            "num_houses": int(len(group)),
            "total_members": int(group["Total Members"].sum()),
            "households": households,
            "risk_probability": prob,
            "is_high_risk": prob >= 50,
            "risk_level": level,
            "recommendation": rec,
        })
    results.sort(key=lambda x: x["risk_probability"], reverse=True)
    return jsonify(results)

@app.route("/api/report")
def get_report():
    zone_data = []
    for ap, group in df.groupby("Accumulation Point"):
        avg_kg = float(group["Garbage Produced (kg)/Month"].mean())
        prob, level, _ = predict_risk(avg_kg)
        zone_data.append({
            "accumulation_point": int(ap),
            "total_garbage": round(float(group["Garbage Produced (kg)/Month"].sum()), 2),
            "avg_garbage": round(avg_kg, 2),
            "min_garbage": round(float(group["Garbage Produced (kg)/Month"].min()), 2),
            "max_garbage": round(float(group["Garbage Produced (kg)/Month"].max()), 2),
            "num_houses": int(len(group)),
            "total_members": int(group["Total Members"].sum()),
            "risk_probability": prob,
            "risk_level": level,
        })

    streets = df["Street"].value_counts().to_dict()
    return jsonify({
        "total_waste_kg": round(float(df["Garbage Produced (kg)/Month"].sum()), 2),
        "total_households": len(df),
        "total_members": int(df["Total Members"].sum()),
        "avg_waste_per_house": round(float(df["Garbage Produced (kg)/Month"].mean()), 2),
        "num_zones": int(df["Accumulation Point"].nunique()),
        "high_risk_zones": sum(1 for z in zone_data if z["risk_probability"] >= 50),
        "streets": [{"name": k, "count": v} for k, v in streets.items()],
        "zone_data": zone_data,
        "model_info": {
            "type": "Logistic Regression",
            "feature": "Avg Garbage (kg/Month) per household",
            "threshold_kg": 12,
            "coef": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }
    })

if __name__ == "__main__":
    print("Plastic Waste Mapper API — http://localhost:5000")
    app.run(debug=True, port=5000)
