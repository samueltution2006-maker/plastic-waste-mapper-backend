import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_excel("survey_csp_veerpuram.xlsx")
df['HighAccumulation'] = (df['Garbage Produced (kg)/Month'] > 12).astype(int)
X = df[['Garbage Produced (kg)/Month']]
y = df['HighAccumulation']
model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "garbage_model.pkl")
print("Model trained and saved as garbage_model.pkl")
