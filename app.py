from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pytesseract
from PIL import Image
import re
import os
import joblib
import uuid
import pandas as pd

# ---------------- TESSERACT PATH ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = joblib.load("diabetes_model.pkl")

# ---------------- LOAD DIET DATASET ----------------
diet_data = pd.read_csv("diet_dataset.csv")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =====================================================
# DYNAMIC DIET FUNCTION (BASED ON YOUR DATASET)
# =====================================================

def generate_structured_diet(days, prediction):

    diet_plan = []

    # Decide group keyword
    if prediction == 1:
        group_keyword = "Diabetic_A"
        exercise = "45 Minutes Walking"
    else:
        group_keyword = "Diabetic_N"
        exercise = "30 Minutes Walking"

    # Filter rows that CONTAIN the group keyword
    filtered = diet_data[diet_data["Group"].str.contains(group_keyword, case=False, na=False)]

    # Clean Meal column (remove spaces)
    filtered["Meal"] = filtered["Meal"].str.strip()

    breakfasts = filtered[filtered["Meal"] == "Breakfast"]
    lunches = filtered[filtered["Meal"] == "Lunch"]
    dinners = filtered[filtered["Meal"] == "Dinner"]

    # Safety check
    if breakfasts.empty or lunches.empty or dinners.empty:
        return [{
            "day": 1,
            "breakfast": "No breakfast data found",
            "lunch": "No lunch data found",
            "dinner": "No dinner data found",
            "exercise": exercise
        }]

    for day in range(1, days + 1):

        breakfast = breakfasts.sample(n=1, replace=True).iloc[0]
        lunch = lunches.sample(n=1, replace=True).iloc[0]
        dinner = dinners.sample(n=1, replace=True).iloc[0]

        diet_plan.append({
            "day": day,
            "breakfast": f"{breakfast['Dish']} ({breakfast['Calories']} kcal)",
            "lunch": f"{lunch['Dish']} ({lunch['Calories']} kcal)",
            "dinner": f"{dinner['Dish']} ({dinner['Calories']} kcal)",
            "exercise": exercise
        })

    return diet_plan



# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html",
                           prediction=None,
                           probability=None,
                           medicine=None,
                           diet=None)


@app.route("/check")
def check():
    return render_template("check.html")


@app.route("/manual")
def manual():
    return render_template("manual.html")


@app.route("/upload_page")
def upload_page():
    return render_template("upload.html")


# =====================================================
# MANUAL PREDICTION
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        def safe_float(value, default):
            try:
                return float(value)
            except:
                return default

        Pregnancies = safe_float(request.form.get("Pregnancies"), 0)
        Glucose = safe_float(request.form.get("Glucose"), 100)
        BloodPressure = safe_float(request.form.get("BloodPressure"), 70)
        SkinThickness = safe_float(request.form.get("SkinThickness"), 20)
        Insulin = safe_float(request.form.get("Insulin"), 80)
        BMI = safe_float(request.form.get("BMI"), 25)
        DPF = safe_float(request.form.get("DiabetesPedigreeFunction"), 0.5)
        Age = safe_float(request.form.get("Age"), 30)

        data = np.array([[Pregnancies, Glucose, BloodPressure,
                          SkinThickness, Insulin, BMI, DPF, Age]])

        prediction = int(model.predict(data)[0])
        probability = round(model.predict_proba(data)[0][1] * 100, 2)

        if prediction == 1:
            medicine = "⚠ High Risk: Consult Diabetologist"
        else:
            medicine = "✅ Low Risk: Maintain Healthy Lifestyle"

        return render_template("dashboard.html",
                               prediction=prediction,
                               probability=probability,
                               medicine=medicine,
                               diet=None)

    except Exception as e:
        return f"Error occurred: {e}"


# =====================================================
# OCR UPLOAD PREDICTION
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return redirect(url_for("dashboard"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("dashboard"))

    unique_name = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    text = pytesseract.image_to_string(Image.open(filepath))

    def extract_value(keyword, default):
        pattern = rf"{keyword}\s*[:\-]?\s*(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else default

    Pregnancies = extract_value("Pregnancies", 0)
    Glucose = extract_value("Glucose", 100)
    BloodPressure = extract_value("Pressure", 70)
    SkinThickness = extract_value("Skin", 20)
    Insulin = extract_value("Insulin", 80)
    BMI = extract_value("BMI", 25)
    DPF = extract_value("DPF", 0.5)
    Age = extract_value("Age", 30)

    data = np.array([[Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI, DPF, Age]])

    prediction = int(model.predict(data)[0])
    probability = round(model.predict_proba(data)[0][1] * 100, 2)

    if prediction == 1:
        medicine = "⚠ High Risk: Consult Diabetologist"
    else:
        medicine = "✅ Low Risk: Maintain Healthy Lifestyle"

    return render_template("dashboard.html",
                           prediction=prediction,
                           probability=probability,
                           medicine=medicine,
                           diet=None)


# =====================================================
# DIET ROUTE
# =====================================================

@app.route("/diet", methods=["POST"])
def diet():

    days = int(request.form.get("days"))
    prediction = int(request.form.get("prediction"))

    diet_plan = generate_structured_diet(days, prediction)

    if prediction == 1:
        medicine = "⚠ High Risk: Consult Diabetologist"
    else:
        medicine = "✅ Low Risk: Maintain Healthy Lifestyle"

    return render_template("dashboard.html",
                           prediction=prediction,
                           probability=None,
                           medicine=medicine,
                           diet=diet_plan)


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
