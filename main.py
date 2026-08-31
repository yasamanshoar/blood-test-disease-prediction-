import sys
import json
import base64
import requests
import joblib
import pandas as pd

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5 import uic
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# CONFIG
# =========================
GEMINI_MODEL = "gemini-3.1-flash-lite"
ML_MODEL_PATH = "blood_disease_pipeline.pkl"


# =========================
# LOAD ML MODEL
# =========================
ml_model = joblib.load(ML_MODEL_PATH)


# =========================
# SAFE JSON PARSER (VERY IMPORTANT)
# =========================
def safe_json_loads(text): 
    """
    پاک‌سازی خروجی Gemini (markdown / متن اضافه) و تبدیل امن به JSON
    """
    text = text.strip()

    # حذف ```json و ```
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    # فقط بخش JSON را نگه دار
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


# =========================
# MAP GEMINI TESTS → ML FEATURES
# =========================
def map_tests_to_features(test_list):
    features = {}

    for item in test_list:
        name = item.get("Test Name", "").lower()
        value = item.get("Result")

        try:
            value = float(value)
        except:
            continue

        if "ferritin" in name:
            features["ferritin"] = value
        elif "hb a1c" in name or "hba1c" in name:
            features["hba1c"] = value
        elif "tsh" in name:
            features["tsh"] = value
        elif "vitamin d" in name:
            features["vitamin_d"] = value
        elif "crp" in name:
            features["crp"] = value
        elif "esr" in name:
            features["esr"] = value
        elif "potassium" in name:
            features["potassium"] = value

    return features


# =========================
# MAIN WINDOW
# =========================
class medical(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("medical.ui", self)

        self.imagesend.clicked.connect(self.on_send_text)
        self.imageadd.clicked.connect(self.on_send_image)
        self.trash.clicked.connect(self.clear_prompt)

    # =========================
    # CLEAR
    # =========================
    def clear_prompt(self):
        self.prompt.clear()
        self.Output.clear()
        self.Output.append("🗑️ حافظه پاک شد\n")
         

    # =========================
    # TEXT → GEMINI ONLY
    # =========================
    def on_send_text(self):
        api_key = self.key3.toPlainText().strip()
        prompt = self.prompt.toPlainText().strip()
        
        if not api_key or not prompt:
            QMessageBox.critical(self, "Error", "Prompt یا API Key خالی است")
            return

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = requests.post(url, params={"key": api_key}, json=data)
        if response.status_code == 200:
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            self.Output.append(text + "\n")
        else:
            self.Output.append(response.text + "\n") 

    # =========================
    # IMAGE FLOW - ENTRY POINT
    # =========================
    def on_send_image(self):
        api_key = self.key3.toPlainText().strip()
        if not api_key:
            QMessageBox.critical(self, "Error", "API Key وارد نشده")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg )"
        )
        if not image_path:
            return

        file_name = image_path.split("/")[-1]
        self.Output.append(f"📁 تصویر انتخاب شده: {file_name}\n")
        self.Output.append("⏳ در حال پردازش...\n")
        QApplication.processEvents()

        # اول سعی می‌کنیم JSON استخراج کنیم (حالت تشخیص)
        success = self.try_extract_json_and_process(image_path, api_key)
        
        # اگر موفق نبودیم، تحلیل مستقیم انجام می‌دهیم
        if not success:
            self.Output.append("⚠️ سیستم نتوانست داده‌های آزمایش را استخراج کند\n")
            self.Output.append("🔄 در حال تحلیل مستقیم تصویر...\n")
            QApplication.processEvents()
            self.direct_image_analysis(image_path, api_key)

    # ========================= 
    # TRY TO EXTRACT JSON (PROCESS IMAGE)
    # =========================
    def try_extract_json_and_process(self, image_path, api_key):
        """
        سعی می‌کند JSON از تصویر استخراج کند و با ML پردازش کند
        برمی‌گرداند: True اگر موفق بود، False اگر شکست خورد
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()

        extract_prompt = """
این تصویر یک برگه آزمایش خون است.
نتایج آزمایش را فقط به صورت JSON (لیست) برگردان.
هر آیتم شامل Test Name و Result باشد.
هیچ متن اضافه‌ای ننویس.
"""

        data = {
            "contents": [{
                "parts": [
                    {"text": extract_prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}
                ]
            }]
        }

        try:
            response = requests.post(url, params={"key": api_key}, json=data, timeout=30)
            if response.status_code != 200:
                return False

            raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            # سعی در تجزیه JSON
            try:
                parsed = safe_json_loads(raw_text)
            except Exception:
                return False

            # بررسی اینکه آیا داده‌های معتبر داریم
            if isinstance(parsed, list):
                extracted_data = map_tests_to_features(parsed)
            else:
                extracted_data = parsed

            if not extracted_data:
                return False

            # اگر به اینجا رسیدیم، موفق بوده‌ایم
            self.Output.append("✅ داده‌های آزمایش با موفقیت استخراج شد\n")
            self.process_with_ml(extracted_data, api_key)
            return True

        except Exception:
            return False

    # =========================
    # PROCESS WITH ML (WHEN EXTRACTION SUCCESSFUL)
    # =========================
    def process_with_ml(self, extracted_data, api_key):
        """
        پردازش داده‌های استخراج شده با مدل ML
        """
        # =========================
        # ML PREDICTION + CONFIDENCE
        # =========================
        df = pd.DataFrame([extracted_data])
        df = df.reindex(columns=ml_model.feature_names_in_, fill_value=0)

        probs = ml_model.predict_proba(df)[0]
        classes = ml_model.classes_

        results = sorted(
            [{"disease": c, "confidence": round(p * 100, 1)}
             for c, p in zip(classes, probs)],
            key=lambda x: x["confidence"],
            reverse=True
        )

        # =========================
        # OUTPUT ML RESULTS
        # =========================
        self.Output.append("="*10 + "\n")
        self.Output.append("🤖 پیش‌بینی مدل یادگیری ماشین:\n")
        for i, r in enumerate(results[:3], 1):
            self.Output.append(f"{i}. {r['disease']} → {r['confidence']}%\n")

        # =========================
        # GET EXPLANATION FROM GEMINI
        # =========================
        explain_prompt = f"""
بر اساس نتایج مدل یادگیری ماشین:

{results[:3]}

لطفاً:
1. بیماری اول را محتمل
2. دومی را متوسط
3. سومی را ضعیف توضیح بده
4. بگو به چه پزشکی مراجعه شود
5. تاکید کن تشخیص قطعی نیست

در هر از سه گزینه در سطر اول نام بیماری در سطر دوم درصد احتمال و در سطر سوم توضیحات را بنویس و در انتها با 10 علامت مساوی هر نتیجه را جدا کن  
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        explain_data = {"contents": [{"parts": [{"text": explain_prompt}]}]}
        
        try:
            explain_response = requests.post(
                url, params={"key": api_key}, json=explain_data, timeout=30
            )

            if explain_response.status_code == 200:
                text = explain_response.json()["candidates"][0]["content"]["parts"][0]["text"]
                self.Output.append("\n🧠 توضیحات:\n")
                self.Output.append(text + "\n")
                self.Output.append("="*10 + "\n")
            else:
                self.Output.append(f"❌ خطا در دریافت توضیحات\n")
        except Exception as e:
            self.Output.append(f"❌ خطا: {str(e)}\n")

    # =========================
    # DIRECT IMAGE ANALYSIS (WHEN EXTRACTION FAILED)
    # =========================
    def direct_image_analysis(self, image_path, api_key):
        """
        تحلیل مستقیم تصویر - وقتی JSON استخراج نمی‌شود
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """بر اساس ازمایش خون داده شده سه احتمال محتمل متوسط و ضعیف را با نمایش درصد بنویس و تاکیید کن تشخیص قطعی نیست و بیمار باید به دکتر مراجعه کند و در هر سه گزینه در سطر اول نام بیماری در سطر دوم درصد احتمال و در سطر سوم توضیحات را بنویس و در نهایت با 10 علامت مساوی هر نتیجه را جدا کن"""

        data = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }]
        }

        try:
            response = requests.post(
                url,
                params={"key": api_key},
                json=data,
                timeout=30
            )

            self.Output.append("="*10 + "\n")
            self.Output.append("🔍 تحلیل مستقیم تصویر:\n")
            self.Output.append("="*10 + "\n")

            if response.status_code == 200:
                js = response.json()
                text = js["candidates"][0]["content"]["parts"][0]["text"]
                self.Output.append(text + "\n")
            else:
                self.Output.append(f"❌ خطا: {response.status_code}\n")
                self.Output.append(response.text + "\n")

        except Exception as e:
            self.Output.append(f"❌ خطا: {str(e)}\n")


# =========================
# MAIN
# =========================
def main():
    app = QApplication(sys.argv)
    win = medical()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()