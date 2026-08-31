import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# =====================
# 1) Load dataset
# =====================
df = pd.read_csv("blood_test_synthetic_dataset.csv")

# =====================
# 2) حذف ستون‌های ID 
# =====================
for col in df.columns:
    if any(k in col.lower() for k in ["id", "patient", "code"]):
        df = df.drop(columns=[col])

# =====================
# 3) جدا کردن X و y
# =====================
y = df["label_disease"]
X = df.drop(columns=["label_disease"])

# =====================
# 4) تبدیل ستون‌های متنی به عدد (Male/Female و هر چیز مشابه)
# =====================
X = pd.get_dummies(X)

# =====================
# 5) اطمینان نهایی (نباید object بماند)
# =====================
print("Remaining dtypes:\n", X.dtypes.value_counts())

# =====================
# 6) Train model
# =====================
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# =====================
# 7) Save trained model
# =====================
joblib.dump(model, "blood_disease_pipeline.pkl")

print("\n✅ Model trained and saved as blood_disease_pipeline.pkl")
