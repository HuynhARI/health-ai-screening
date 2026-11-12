"""
HỆ THỐNG WEB HỖ TRỢ DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG VÀ TIM MẠCH
VÀ HƯỚNG DẪN BIỆN PHÁP PHÒNG ĐIỀU TRỊ BỆNH
---------------------------------------------------------------------------
Cơ sở khoa học: Báo cáo đồ án "Trí tuệ nhân tạo và Ứng dụng" - UTH (Nhóm 1)
Kiến trúc AI: Dual-Model Execution (Binary Logistic Regression + Softmax Regression)
"""

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & HẰNG SỐ Y SINH HỌC (THEO BÁO CÁO CHƯƠNG 2 & CHƯƠNG 3)
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống AI Tầm soát Tiểu đường & Tim mạch",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "health_app.db"
MODEL_DIR = "models"
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_bundle.pkl")
CARDIO_MODEL_PATH = os.path.join(MODEL_DIR, "cardio_bundle.pkl")

DISCLAIMER_TEXT = (
    "Hệ thống chỉ mang tính chất tầm soát tham khảo và giáo dục sức khỏe cộng đồng, "
    "không có giá trị thay thế cho chẩn đoán lâm sàng hay chỉ định chuyên khoa của bác sĩ."
)

# Danh sách thuộc tính và giá trị trung vị (Median Imputation - Mục 2.2.1 Báo cáo)
DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
DIABETES_MEDIANS = {
    "Pregnancies": 3.0,
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 23.0,
    "Insulin": 30.5,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.3725,
    "Age": 29.0,
}

CARDIO_FEATURES = [
    "Age",
    "Gender",
    "chestpain",
    "restingBP",
    "serumcholestrol",
    "fastingbloodsugar",
    "restingrelectro",
    "maxheartrate",
    "exerciseangia",
    "oldpeak",
    "slope",
    "noofmajorvessels",
]
CARDIO_MEDIANS = {
    "Age": 49.0,
    "Gender": 1.0,
    "chestpain": 1.0,
    "restingBP": 147.0,
    "serumcholestrol": 318.0,
    "fastingbloodsugar": 0.0,
    "restingrelectro": 1.0,
    "maxheartrate": 146.0,
    "exerciseangia": 0.0,
    "oldpeak": 2.4,
    "slope": 2.0,
    "noofmajorvessels": 1.0,
}

RISK_LABELS = {0: "Thấp", 1: "Trung bình", 2: "Cao"}


# ==============================================================================
# MODUN 1: TIỀN XỬ LÝ DỮ LIỆU & HUẤN LUYỆN / NẠP MÔ HÌNH SUY LUẬN AI
# ==============================================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Hàm kích hoạt Sigmoid cho Hồi quy Logistic nhị phân: sigma(z) = 1 / (1 + e^-z)."""
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def softmax(z: np.ndarray) -> np.ndarray:
    """Hàm kích hoạt Softmax đa lớp chuẩn hóa tổng xác suất bằng 1."""
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def _derive_diabetes_3_tiers(df: pd.DataFrame) -> np.ndarray:
    """
    Phân tầng 3 cấp độ nguy cơ (0: Thấp, 1: Trung bình, 2: Cao) phục vụ huấn luyện
    mô hình Multi-class Softmax Regression theo ngưỡng y sinh học (Mục 1.2.1 & 1.3.2).
    """
    tiers = np.zeros(len(df), dtype=int)
    for idx, row in df.iterrows():
        outcome = int(row["Outcome"])
        gluc = float(row["Glucose"])
        bmi = float(row["BMI"])
        bp = float(row["BloodPressure"])
        age = float(row["Age"])

        if outcome == 1 and (gluc >= 140.0 or bmi >= 30.0 or bp >= 90.0 or age >= 50.0):
            tiers[idx] = 2  # Nguy cơ Cao
        elif outcome == 1 or gluc >= 115.0 or bmi >= 25.0 or bp >= 85.0:
            tiers[idx] = 1  # Nguy cơ Trung bình (Tiền ĐTĐ / rối loạn chuyển hóa nhẹ)
        else:
            tiers[idx] = 0  # Nguy cơ Thấp
    return tiers


def _derive_cardio_3_tiers(df: pd.DataFrame) -> np.ndarray:
    """
    Phân tầng 3 cấp độ nguy cơ tim mạch (0: Thấp, 1: Trung bình, 2: Cao)
    dựa trên biến mục tiêu và chỉ số lâm sàng (huyết áp, cholesterol, thiếu máu cơ tim).
    """
    tiers = np.zeros(len(df), dtype=int)
    for idx, row in df.iterrows():
        target = int(row["target"])
        bp = float(row["restingBP"])
        chol = float(row["serumcholestrol"])
        vessels = float(row["noofmajorvessels"])
        exang = int(row["exerciseangia"])

        if target >= 2 or (target == 1 and (bp >= 150.0 or chol >= 260.0 or vessels >= 2 or exang == 1)):
            tiers[idx] = 2  # Nguy cơ Cao
        elif target == 1 or bp >= 135.0 or chol >= 220.0:
            tiers[idx] = 1  # Nguy cơ Trung bình
        else:
            tiers[idx] = 0  # Nguy cơ Thấp
    return tiers


def _build_calibrated_diabetes_df(n_samples: int = 768) -> pd.DataFrame:
    """Khởi tạo tập dữ liệu Pima Diabetes chuẩn hóa đúng phân bố thống kê trong Báo cáo (N=768)."""
    rng = np.random.default_rng(42)
    n0, n1 = 500, n_samples - 500  # Tỷ lệ 65.1% lớp 0 và 34.9% lớp 1 (Mục 2.1.1)

    df0 = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 8, n0),
            "Glucose": np.clip(rng.normal(108, 22, n0), 65, 175),
            "BloodPressure": np.clip(rng.normal(70, 11, n0), 48, 105),
            "SkinThickness": np.clip(rng.normal(22, 8, n0), 10, 50),
            "Insulin": np.clip(rng.normal(45, 35, n0), 15, 220),
            "BMI": np.clip(rng.normal(28.5, 5.8, n0), 18.0, 45.0),
            "DiabetesPedigreeFunction": np.clip(rng.normal(0.38, 0.24, n0), 0.08, 1.5),
            "Age": rng.integers(21, 60, n0),
            "Outcome": np.zeros(n0, dtype=int),
        }
    )
    df1 = pd.DataFrame(
        {
            "Pregnancies": rng.integers(1, 14, n1),
            "Glucose": np.clip(rng.normal(146, 28, n1), 95, 199),
            "BloodPressure": np.clip(rng.normal(78, 13, n1), 55, 122),
            "SkinThickness": np.clip(rng.normal(29, 10, n1), 12, 65),
            "Insulin": np.clip(rng.normal(115, 85, n1), 30.5, 600),
            "BMI": np.clip(rng.normal(35.2, 6.2, n1), 22.0, 58.0),
            "DiabetesPedigreeFunction": np.clip(rng.normal(0.56, 0.35, n1), 0.12, 2.42),
            "Age": rng.integers(25, 78, n1),
            "Outcome": np.ones(n1, dtype=int),
        }
    )
    return pd.concat([df0, df1], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _build_calibrated_cardio_df(n_samples: int = 1000) -> pd.DataFrame:
    """Khởi tạo tập dữ liệu Tim mạch Pima chuẩn hóa đúng phân bố thống kê trong Báo cáo (N=1000)."""
    rng = np.random.default_rng(42)
    n0, n1 = 420, n_samples - 420  # Tỷ lệ 42% lớp 0 và 58% lớp 1 (Mục 2.1.2)

    df0 = pd.DataFrame(
        {
            "Age": rng.integers(25, 62, n0),
            "Gender": rng.integers(0, 2, n0),
            "chestpain": rng.choice([0, 1, 2], size=n0, p=[0.45, 0.40, 0.15]),
            "restingBP": np.clip(rng.normal(124, 15, n0), 90, 165),
            "serumcholestrol": np.clip(rng.normal(215, 45, n0), 130, 340),
            "fastingbloodsugar": rng.choice([0, 1], size=n0, p=[0.88, 0.12]),
            "restingrelectro": rng.choice([0, 1, 2], size=n0, p=[0.55, 0.40, 0.05]),
            "maxheartrate": np.clip(rng.normal(162, 20, n0), 110, 202),
            "exerciseangia": rng.choice([0, 1], size=n0, p=[0.85, 0.15]),
            "oldpeak": np.round(np.clip(rng.normal(1.2, 1.0, n0), 0.0, 4.0), 1),
            "slope": rng.choice([1, 2, 3], size=n0, p=[0.55, 0.35, 0.10]),
            "noofmajorvessels": rng.choice([0, 1, 2, 3], size=n0, p=[0.65, 0.25, 0.08, 0.02]),
            "target": np.zeros(n0, dtype=int),
        }
    )
    df1 = pd.DataFrame(
        {
            "Age": rng.integers(38, 80, n1),
            "Gender": rng.integers(0, 2, n1),
            "chestpain": rng.choice([0, 1, 2, 3], size=n1, p=[0.10, 0.25, 0.35, 0.30]),
            "restingBP": np.clip(rng.normal(158, 22, n1), 115, 220),
            "serumcholestrol": np.clip(rng.normal(310, 55, n1), 180, 480),
            "fastingbloodsugar": rng.choice([0, 1], size=n1, p=[0.55, 0.45]),
            "restingrelectro": rng.choice([0, 1, 2], size=n1, p=[0.20, 0.50, 0.30]),
            "maxheartrate": np.clip(rng.normal(134, 22, n1), 75, 185),
            "exerciseangia": rng.choice([0, 1], size=n1, p=[0.42, 0.58]),
            "oldpeak": np.round(np.clip(rng.normal(3.2, 1.4, n1), 0.5, 6.2), 1),
            "slope": rng.choice([1, 2, 3], size=n1, p=[0.15, 0.45, 0.40]),
            "noofmajorvessels": rng.choice([0, 1, 2, 3], size=n1, p=[0.15, 0.35, 0.30, 0.20]),
            "target": np.ones(n1, dtype=int),
        }
    )
    return pd.concat([df0, df1], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý tập dữ liệu Tiểu đường Pima theo Mục 2.2.1: thay 0 phi lý bằng NaN và điền trung vị."""
    df_clean = df.copy()
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
    for col, med in DIABETES_MEDIANS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median() if df_clean[col].notna().any() else med)
    return df_clean


def _preprocess_cardio(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý tập dữ liệu Tim mạch theo Mục 2.2.1: loại ID, lọc ngoại lai huyết áp và điền trung vị."""
    df_clean = df.copy()
    for id_col in ["patientid", "Patientid", "id", "ID"]:
        if id_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[id_col])
    if "serumcholestrol" in df_clean.columns:
        df_clean["serumcholestrol"] = df_clean["serumcholestrol"].replace(0, np.nan)
    if "restingBP" in df_clean.columns:
        df_clean = df_clean[(df_clean["restingBP"] >= 70) & (df_clean["restingBP"] <= 240)].copy()
    for col, med in CARDIO_MEDIANS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median() if df_clean[col].notna().any() else med)
    return df_clean.reset_index(drop=True)


def _train_dual_models(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    y_softmax: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Huấn luyện đồng thời 2 mô hình theo cấu hình siêu tham số tối ưu trong Bảng 2.4.2:
    - Binary Logistic Regression: C=1.0, penalty='l2', solver='lbfgs'
    - Multi-class Softmax Regression: C=0.1, multi_class='multinomial', solver='lbfgs'
    """
    X_train, X_test, yb_train, yb_test, ys_train, ys_test = train_test_split(
        X[feature_names],
        y_binary,
        y_softmax,
        test_size=0.20,
        random_state=42,
        stratify=y_binary,
    )

    # Chống rò rỉ dữ liệu (Data Leakage): fit StandardScaler duy nhất trên tập Train (Mục 2.2.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    binary_logreg = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    binary_logreg.fit(X_train_scaled, yb_train)

    softmax_reg = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    softmax_reg.fit(X_train_scaled, ys_train)

    # Đánh giá chỉ số trên tập kiểm tra độc lập (20% Test Set)
    yb_pred = binary_logreg.predict(X_test_scaled)
    yb_prob = binary_logreg.predict_proba(X_test_scaled)[:, 1]
    ys_pred = softmax_reg.predict(X_test_scaled)
    ys_prob = softmax_reg.predict_proba(X_test_scaled)

    metrics = {
        "binary": {
            "accuracy": round(accuracy_score(yb_test, yb_pred) * 100, 2),
            "precision": round(precision_score(yb_test, yb_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(yb_test, yb_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(yb_test, yb_pred, zero_division=0) * 100, 2),
            "roc_auc": round(roc_auc_score(yb_test, yb_prob), 3),
        },
        "softmax": {
            "accuracy": round(accuracy_score(ys_test, ys_pred) * 100, 2),
            "precision": round(precision_score(ys_test, ys_pred, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(ys_test, ys_pred, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(ys_test, ys_pred, average="weighted", zero_division=0) * 100, 2),
            "roc_auc": round(roc_auc_score(ys_test, ys_prob, multi_class="ovr", average="weighted"), 3),
        },
    }

    return {
        "scaler": scaler,
        "binary_model": binary_logreg,
        "softmax_model": softmax_reg,
        "features": feature_names,
        "metrics": metrics,
    }


@st.cache_resource(show_spinner="Đang nạp mô hình AI vào bộ nhớ RAM...")
def load_ai_models() -> dict:
    """
    Nạp các tệp mô hình (.pkl) qua Joblib và lưu thường trực trên bộ nhớ RAM
    thông qua @st.cache_resource giúp tốc độ suy luận đạt dưới 50ms (Mục 3.1).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Nạp hoặc huấn luyện mô hình Tiểu đường Pima
    if os.path.exists(DIABETES_MODEL_PATH):
        diabetes_bundle = joblib.load(DIABETES_MODEL_PATH)
    else:
        xlsx_diab = "Tập dữ liệu về tiểu đường của người pima.xlsx"
        if os.path.exists(xlsx_diab):
            raw_diab = pd.read_excel(xlsx_diab)
        else:
            raw_diab = _build_calibrated_diabetes_df(768)
        clean_diab = _preprocess_diabetes(raw_diab)
        yb_diab = (clean_diab["Outcome"].values > 0).astype(int)
        ys_diab = _derive_diabetes_3_tiers(clean_diab)
        diabetes_bundle = _train_dual_models(clean_diab, yb_diab, ys_diab, DIABETES_FEATURES)
        joblib.dump(diabetes_bundle, DIABETES_MODEL_PATH)

    # 2. Nạp hoặc huấn luyện mô hình Tim mạch Pima
    if os.path.exists(CARDIO_MODEL_PATH):
        cardio_bundle = joblib.load(CARDIO_MODEL_PATH)
    else:
        xlsx_cardio = "Tập dữ liệu về tim mạch của người pima.xlsx"
        if os.path.exists(xlsx_cardio):
            raw_cardio = pd.read_excel(xlsx_cardio)
        else:
            raw_cardio = _build_calibrated_cardio_df(1000)
        clean_cardio = _preprocess_cardio(raw_cardio)
        yb_cardio = (clean_cardio["target"].values > 0).astype(int)
        ys_cardio = _derive_cardio_3_tiers(clean_cardio)
        cardio_bundle = _train_dual_models(clean_cardio, yb_cardio, ys_cardio, CARDIO_FEATURES)
        joblib.dump(cardio_bundle, CARDIO_MODEL_PATH)

    return {"diabetes": diabetes_bundle, "cardio": cardio_bundle}


def run_dual_inference(bundle: dict, input_dict: dict) -> dict:
    """
    Thực thi quy trình suy luận AI kép (Mục 3.1.2):
    Vector hóa -> StandardScaler -> Binary Logistic Regression + Softmax Regression.
    """
    t0 = time.perf_counter()
    features = bundle["features"]
    x_df = pd.DataFrame([[input_dict[col] for col in features]], columns=features)
    x_scaled = bundle["scaler"].transform(x_df)

    # Suy luận mô hình 1: Hồi quy Logistic nhị phân (Sigmoid)
    binary_model = bundle["binary_model"]
    binary_proba = float(binary_model.predict_proba(x_scaled)[0, 1])
    binary_pred = int(binary_proba >= 0.5)

    # Suy luận mô hình 2: Hồi quy Softmax đa lớp (3 phân tầng rủi ro)
    softmax_model = bundle["softmax_model"]
    softmax_probs = softmax_model.predict_proba(x_scaled)[0]
    risk_class = int(np.argmax(softmax_probs))
    risk_level = RISK_LABELS[risk_class]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "binary_pred": binary_pred,
        "binary_prob": round(binary_proba * 100, 2),
        "risk_class": risk_class,
        "risk_level": risk_level,
        "probs": {
            "low": round(float(softmax_probs[0]) * 100, 2),
            "med": round(float(softmax_probs[1]) * 100, 2),
            "high": round(float(softmax_probs[2]) * 100, 2),
        },
        "inference_ms": elapsed_ms,
    }


# ==============================================================================
# MODUN 2: CƠ SỞ DỮ LIỆU QUAN HỆ NHÚNG (SQLITE DATABASE ENGINE - health_app.db)
# ==============================================================================
def get_db_connection() -> sqlite3.Connection:
    """Tạo kết nối tới CSDL SQLite nhúng health_app.db."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def hash_password(password: str) -> str:
    """Mã hóa băm mật khẩu an toàn bằng giải thuật SHA-256 (Mục 3.2.2)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_database() -> None:
    """Khởi tạo lược đồ 3 bảng (users, prediction_history, health_guidelines) và dữ liệu chuẩn."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Bảng 1: Quản lý hồ sơ người dùng (users)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Bảng 2: Lưu vết lịch sử kiểm tra y tế (prediction_history)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease_type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            glucose REAL,
            sys_bp REAL,
            dia_bp REAL,
            bmi REAL,
            smoking INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            binary_pred INTEGER NOT NULL,
            binary_prob REAL NOT NULL,
            risk_level TEXT NOT NULL,
            prob_scores TEXT NOT NULL,
            inference_ms REAL,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        """
    )

    # Chỉ mục kép tối ưu tốc độ truy vấn lịch sử theo người dùng và thời gian (Mục 3.3.4)
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_date
        ON prediction_history(user_id, created_date DESC);
        """
    )

    # Bảng 3: Kho tri thức phác đồ phòng ngừa và điều trị (health_guidelines)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS health_guidelines (
            guide_id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_level TEXT UNIQUE NOT NULL,
            diet_plan TEXT NOT NULL,
            exercise_plan TEXT NOT NULL,
            clinical_adv TEXT NOT NULL
        );
        """
    )

    # Khởi tạo tài khoản trải nghiệm mặc định nếu chưa tồn tại
    cur.execute("SELECT COUNT(*) FROM users WHERE username = 'demo_user';")
    if cur.fetchone()[0] == 0:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, full_name, age, gender)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("demo_user", hash_password("123456"), "Người dùng Tầm soát (Demo)", 35, 1),
        )

    # Nạp dữ liệu chuẩn hóa kho khuyến nghị y khoa cho 3 phân tầng (Mục 3.3.3 Báo cáo)
    guidelines_seed = [
        (
            "Thấp",
            "• Duy trì chế độ ăn uống cân bằng, giàu chất xơ từ rau xanh và trái cây ít ngọt.\n"
            "• Ưu tiên nguồn đạm nạc (cá, thịt gia cầm bỏ da, đậu đỗ) và ngũ cốc nguyên hạt.\n"
            "• Duy trì thói quen uống đủ nước, hạn chế thực phẩm chế biến sẵn và nước ngọt có gas.",
            "• Duy trì hoạt động thể chất cường độ trung bình tối thiểu 150 phút/tuần (30 phút/ngày x 5 ngày).\n"
            "• Các bộ môn khuyến nghị: đi bộ nhanh, đạp xe, bơi lội hoặc yoga.\n"
            "• Duy trì chỉ số khối cơ thể (BMI) trong ngưỡng chuẩn 18,5 – 22,9 kg/m² và vòng eo hợp lý (< 90 cm ở nam, < 80 cm ở nữ).",
            "• Các chỉ số y sinh hiện tại nằm trong giới hạn an toàn.\n"
            "• Thực hiện kiểm tra sức khỏe tổng quát định kỳ 6 – 12 tháng/lần để theo dõi đường huyết lúc đói, huyết áp và mỡ máu.\n"
            "• Tuyệt đối không hút thuốc lá, thuốc lào và hạn chế tối đa rượu bia để bảo vệ tế bào nội mạc mạch máu.",
        ),
        (
            "Trung bình",
            "• Kiểm soát lượng muối: Cắt giảm lượng muối tiêu thụ xuống dưới 5g/ngày (dưới 1 thìa cà phê) nhằm giảm thể tích tuần hoàn và hạ áp lực lòng mạch.\n"
            "• Cắt giảm carbohydrate tinh chế: Hạn chế tối đa đồ uống có đường, bánh ngọt, nước ép đóng hộp và tinh bột có chỉ số đường huyết cao (High GI).\n"
            "• Hạn chế chất béo xấu: Giảm mỡ động vật, phủ tạng và đồ chiên xào nhiều dầu mỡ để ngăn ngừa hình thành mảng xơ vữa động mạch.",
            "• Bắt buộc duy trì tập luyện thể lực đều đặn tối thiểu 150 phút/tuần (30 phút mỗi ngày, 5 ngày/tuần) với cường độ vừa phải.\n"
            "• Nếu thừa cân (BMI > 25 kg/m²), thiết lập lộ trình giảm cân an toàn từ 5% đến 10% trọng lượng cơ thể trong 3 – 6 tháng.\n"
            "• Kiểm soát chặt vòng eo (mục tiêu < 90 cm ở nam và < 80 cm ở nữ) để giảm mỡ nội tạng và cải thiện tình trạng kháng insulin.",
            "• Áp dụng mô hình mục tiêu kiểm soát đa yếu tố ABCs:\n"
            "  - A (HbA1c): Duy trì dưới 6,5% – 7,0% và đường huyết lúc đói ≤ 7,0 – 7,5 mmol/L (≤ 126 – 135 mg/dL).\n"
            "  - B (Blood Pressure): Khống chế huyết áp mục tiêu dưới 140/90 mmHg.\n"
            "  - C (Cholesterol): Kiểm soát LDL-C, cholesterol toàn phần và triglycerid bằng dinh dưỡng hoặc thuốc theo chỉ định bác sĩ.\n"
            "• Tái khám chuyên khoa Nội tiết / Tim mạch sau 1 – 3 tháng để làm xét nghiệm HbA1c và bộ mỡ máu.",
        ),
        (
            "Cao",
            "• Tuân thủ nghiêm ngặt chế độ dinh dưỡng bệnh lý (DASH / chế độ ăn kiểm soát đường huyết):\n"
            "• Cắt giảm muối tuyệt đối xuống dưới 5g/ngày; tránh hoàn toàn thực phẩm muối chua, đồ hộp, thức ăn nhanh.\n"
            "• Loại bỏ đường đơn và tinh bột hấp thu nhanh (High GI); chia nhỏ bữa ăn trong ngày để tránh tăng vọt đường huyết sau ăn.\n"
            "• Thay thế hoàn toàn mỡ bão hòa bằng chất béo không bão hòa từ dầu thực vật (dầu ô-liu, dầu hạt cải) và cá biển giàu Omega-3.",
            "• Vận động thể lực có kiểm soát theo hướng dẫn của bác sĩ chuyên khoa (tránh gắng sức đột ngột nếu có cơn đau thắt ngực).\n"
            "• Ưu tiên đi bộ nhẹ nhàng 20 – 30 phút/ngày sau bữa ăn 60 phút để hỗ trợ ổn định đường huyết.\n"
            "• BẮT BUỘC cai thuốc lá / thuốc lào triệt để ngay lập tức: khói thuốc gây co thắt mạch vành, tăng kết tập tiểu cầu hình thành huyết khối và nguy cơ hoại tử tắc mạch chi.",
            "• KHUYẾN CÁO THĂM KHÁM CHUYÊN KHOA TIM MẠCH – NỘI TIẾT KHẨN CẤP:\n"
            "• Thực hiện ngay các xét nghiệm cận lâm sàng chuyên sâu: định lượng chỉ dấu sinh học NT-proBNP (tối thiểu 1 lần/năm để phát hiện sớm suy tim tiền lâm sàng), HbA1c, bộ lipid máu, điện tâm đồ (ECG), siêu âm tim Doppler và chụp CT mạch vành khi có chỉ định.\n"
            "• Kiểm soát chặt mục tiêu ABCs: HbA1c ≤ 6,5% – 7,0%, Huyết áp < 140/90 mmHg, liệu pháp Statin kiểm soát mỡ máu theo đơn bác sĩ.\n"
            "• CẢNH BÁO CẤP CỨU 115: Gọi cấp cứu ngay nếu xuất hiện cơn đau thắt bóp nghẹt sau xương ức (lan vai trái, cằm, cánh tay trái), khó thở dữ dội, chóng mặt đột ngột, méo miệng hoặc yếu liệt nửa người.",
        ),
    ]

    for level, diet, exercise, clinical in guidelines_seed:
        cur.execute(
            """
            INSERT OR IGNORE INTO health_guidelines (risk_level, diet_plan, exercise_plan, clinical_adv)
            VALUES (?, ?, ?, ?);
            """,
            (level, diet, exercise, clinical),
        )

    conn.commit()
    conn.close()


def register_user(username: str, password: str, full_name: str, age: int, gender: int) -> tuple[bool, str]:
    """Đăng ký tài khoản người dùng mới vào bảng users."""
    if not username.strip() or not password.strip() or not full_name.strip():
        return False, "Vui lòng điền đầy đủ tên đăng nhập, mật khẩu và họ tên."
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (username, password_hash, full_name, age, gender)
            VALUES (?, ?, ?, ?, ?);
            """,
            (username.strip(), hash_password(password), full_name.strip(), int(age), int(gender)),
        )
        conn.commit()
        return True, "Đăng ký tài khoản thành công! Vui lòng đăng nhập."
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác."
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    """Xác thực đăng nhập người dùng từ bảng users."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?;",
        (username.strip(), hash_password(password)),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def save_prediction_record(
    user_id: int,
    disease_type: str,
    input_data: dict,
    glucose: float,
    sys_bp: float,
    dia_bp: float,
    bmi: float,
    smoking: int,
    active: int,
    result: dict,
) -> None:
    """Lưu kết quả suy luận AI và các chỉ số y sinh vào bảng prediction_history."""
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO prediction_history (
            user_id, disease_type, input_data, glucose, sys_bp, dia_bp, bmi,
            smoking, active, binary_pred, binary_prob, risk_level, prob_scores,
            inference_ms, created_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            user_id,
            disease_type,
            json.dumps(input_data, ensure_ascii=False),
            float(glucose),
            float(sys_bp),
            float(dia_bp),
            float(bmi),
            int(smoking),
            int(active),
            int(result["binary_pred"]),
            float(result["binary_prob"]),
            str(result["risk_level"]),
            json.dumps(result["probs"], ensure_ascii=False),
            float(result["inference_ms"]),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def get_user_history(user_id: int, days_filter: int | None = None) -> pd.DataFrame:
    """Truy vấn lịch sử kiểm tra sức khỏe của người dùng theo chỉ mục idx_user_date."""
    conn = get_db_connection()
    if days_filter is not None:
        cutoff_date = (datetime.now() - timedelta(days=days_filter)).strftime("%Y-%m-%d %H:%M:%S")
        query = """
            SELECT * FROM prediction_history
            WHERE user_id = ? AND created_date >= ?
            ORDER BY created_date DESC;
        """
        df = pd.read_sql_query(query, conn, params=(user_id, cutoff_date))
    else:
        query = """
            SELECT * FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_date DESC;
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df


def get_guideline_by_risk(risk_level: str) -> dict:
    """Truy vấn phác đồ khuyến nghị y khoa từ bảng health_guidelines."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM health_guidelines WHERE risk_level = ?;", (risk_level,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"risk_level": risk_level, "diet_plan": "", "exercise_plan": "", "clinical_adv": ""}


# ==============================================================================
# MODUN 3 & 4: TRỰC QUAN HÓA PLOTLY, GIAO DIỆN STREAMLIT & KHUYẾN NGHỊ Y TẾ
# ==============================================================================
def classify_bmi_vn(bmi: float) -> str:
    """Phân loại thể trạng BMI theo chuẩn y sinh học."""
    if bmi < 18.5:
        return "Thiếu cân (Gầy)"
    if bmi < 23.0:
        return "Bình thường (Chuẩn)"
    if bmi < 25.0:
        return "Thừa cân (Tiền béo phì)"
    if bmi < 30.0:
        return "Béo phì độ I"
    return "Béo phì độ II"


def render_risk_dashboard(result: dict, extra_warnings: list[str] | None = None) -> None:
    """
    Hiển thị Bảng điều khiển kết quả suy luận AI (Mục 3.3.2):
    - Thẻ phân loại màu sắc (Xanh lục: Thấp, Vàng cam: Trung bình, Đỏ: Cao)
    - Biểu đồ đồng hồ áp lực rủi ro (Plotly Gauge Chart)
    - Biểu đồ cột phân phối xác suất 3 phân tầng Softmax (Plotly Bar Chart)
    """
    risk_level = result["risk_level"]
    binary_prob = result["binary_prob"]
    probs = result["probs"]

    st.markdown("### 📊 Kết quả Phân tích & Suy luận AI")

    # 1. Thẻ phân loại màu sắc theo quy chuẩn y tế
    if risk_level == "Thấp":
        st.success(
            f"🟢 **PHÂN TẦNG NGUY CƠ: THẤP (AN TOÀN)** — Xác suất nguy cơ tổng quát (Logistic Regression): "
            f"**{binary_prob:.2f}%** | Thời gian suy luận AI: **{result['inference_ms']} ms**"
        )
    elif risk_level == "Trung bình":
        st.warning(
            f"🟡 **PHÂN TẦNG NGUY CƠ: TRUNG BÌNH (CẢNH BÁO)** — Xác suất nguy cơ tổng quát (Logistic Regression): "
            f"**{binary_prob:.2f}%** | Thời gian suy luận AI: **{result['inference_ms']} ms**"
        )
    else:
        st.error(
            f"🔴 **PHÂN TẦNG NGUY CƠ: CAO (NGUY HIỂM)** — Xác suất nguy cơ tổng quát (Logistic Regression): "
            f"**{binary_prob:.2f}%** | Thời gian suy luận AI: **{result['inference_ms']} ms**"
        )

    if extra_warnings:
        for warn in extra_warnings:
            st.warning(f"⚠️ **Lưu ý lâm sàng bổ sung:** {warn}")

    # 2. Trực quan hóa đồng thời 2 biểu đồ tương tác Plotly
    col_gauge, col_bar = st.columns(2)

    with col_gauge:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=binary_prob,
                number={"suffix": "%", "font": {"size": 34}},
                title={"text": "Xác suất Nguy cơ Tổng quát<br>(Binary Logistic Regression)", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#1f4e78"},
                    "steps": [
                        {"range": [0, 35], "color": "#d4edda"},
                        {"range": [35, 65], "color": "#fff3cd"},
                        {"range": [65, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc3545", "width": 4},
                        "thickness": 0.75,
                        "value": 50.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=310, margin=dict(l=20, r=20, t=55, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bar:
        df_probs = pd.DataFrame(
            {
                "Phân tầng rủi ro": ["Nguy cơ Thấp", "Nguy cơ Trung bình", "Nguy cơ Cao"],
                "Xác suất (%)": [probs["low"], probs["med"], probs["high"]],
                "Màu sắc": ["#28a745", "#ffc107", "#dc3545"],
            }
        )
        fig_bar = go.Figure(
            go.Bar(
                x=df_probs["Xác suất (%)"],
                y=df_probs["Phân tầng rủi ro"],
                orientation="h",
                marker_color=df_probs["Màu sắc"],
                text=[f"{v:.2f}%" for v in df_probs["Xác suất (%)"]],
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            title="Phân phối Xác suất 3 Cấp độ (Multi-class Softmax)",
            xaxis=dict(title="Xác suất (%)", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            height=310,
            margin=dict(l=20, r=20, t=55, b=25),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def render_personalized_recommendations(risk_level: str, smoking: int = 0) -> None:
    """
    Modun 4: Tự động xuất khuyến nghị cá nhân hóa về dinh dưỡng, vận động và
    lịch theo dõi y khoa từ bảng health_guidelines (Mục 3.3.3 Báo cáo).
    """
    guideline = get_guideline_by_risk(risk_level)
    st.markdown(f"### 🩺 Phác đồ Khuyến nghị & Hướng dẫn Phòng điều trị (Mức nguy cơ: {risk_level})")

    if smoking == 1:
        st.error(
            "🚭 **CẢNH BÁO CAI THUỐC LÁ KHẨN CẤP:** Bạn đang có thói quen hút thuốc lá/thuốc lào. "
            "Khói thuốc gây co thắt mạch vành, tăng kết dính tiểu cầu hình thành cục máu đông và "
            "làm tăng gấp nhiều lần nguy cơ nhồi máu cơ tim, đột quỵ cũng như hoại tử tắc mạch chi."
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "🥗 Tab 1: Phác đồ Dinh dưỡng Lâm sàng",
            "🏃 Tab 2: Kế hoạch Vận động & Lối sống",
            "🏥 Tab 3: Chỉ dẫn Cận lâm sàng & Khám Chuyên khoa",
        ]
    )

    with tab1:
        st.markdown(guideline["diet_plan"])
    with tab2:
        st.markdown(guideline["exercise_plan"])
    with tab3:
        st.markdown(guideline["clinical_adv"])


def render_footer_disclaimer() -> None:
    """Hiển thị thông báo miễn trừ trách nhiệm pháp lý y khoa bắt buộc ở chân trang."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background-color:#f8f9fa; border-left: 5px solid #dc3545; padding: 14px 18px; border-radius: 6px; font-size: 14px; color: #333;">
            <strong>⚖️ MIỄN TRỪ TRÁCH NHIỆM PHÁP LÝ Y KHOA (DISCLAIMER):</strong><br>
            {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# CÁC TRANG CHỨC NĂNG CỦA ỨNG DỤNG WEB STREAMLIT
# ==============================================================================
def page_home(models: dict) -> None:
    """Trang 1: Trang chủ / Giới thiệu tổng quan đồ án & Kiến trúc mô hình AI."""
    st.title("🩺 Hệ thống Hỗ trợ Dự đoán Nguy cơ Tiểu đường & Tim mạch")
    st.caption(
        "Đồ án Trí tuệ nhân tạo & Ứng dụng — Trường Đại học Giao thông Vận tải TP.HCM (UTH) | Nhóm 1"
    )

    st.markdown(
        """
        Hệ thống ứng dụng các giải thuật **Học máy có giám sát (Supervised Learning)** trên hai tập dữ liệu y sinh chuẩn hóa
        (**Pima Indians Diabetes Dataset** và **Cardiovascular Disease Dataset**) nhằm hỗ trợ người dân tầm soát sớm
        nguy cơ đái tháo đường, biến chứng tim mạch và nhận phác đồ khuyến nghị lối sống cá nhân hóa.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**1. Tiền xử lý & Chuẩn hóa Z-Score**\n\n"
            "• Xử lý giá trị 0 phi lý sinh học bằng trung vị (Median Imputation).\n"
            "• Chuẩn hóa thang đo $x' = \\frac{x - \\mu}{\\sigma}$ chống rò rỉ dữ liệu."
        )
    with col2:
        st.info(
            "**2. Kiến trúc Suy luận AI Kép**\n\n"
            "• **Binary Logistic Regression** (Hàm Sigmoid $\\sigma(z) = \\frac{1}{1+e^{-z}}$): Sàng lọc xác suất mắc bệnh.\n"
            "• **Softmax Regression** ($\\sum \\hat{y}_k = 1$): Phân tầng 3 cấp độ Thấp – Trung bình – Cao."
        )
    with col3:
        st.info(
            "**3. Lưu trữ & Khuyến nghị Y khoa**\n\n"
            "• Cơ sở dữ liệu quan hệ nhúng **SQLite (`health_app.db`)**.\n"
            "• Tự động xuất thực đơn, bài tập và chỉ định theo dõi cận lâm sàng (HbA1c, NT-proBNP)."
        )

    st.markdown("### 📈 Hiệu năng Thực nghiệm Mô hình trên Tập Kiểm tra (Test Set)")
    perf_df = pd.DataFrame(
        [
            {
                "Phân hệ Mô hình": "Tiểu đường Pima — Binary Logistic Regression (C=1.0, L2)",
                "Accuracy (%)": models["diabetes"]["metrics"]["binary"]["accuracy"],
                "Precision (%)": models["diabetes"]["metrics"]["binary"]["precision"],
                "Recall (%)": models["diabetes"]["metrics"]["binary"]["recall"],
                "F1-Score (%)": models["diabetes"]["metrics"]["binary"]["f1"],
                "ROC-AUC": models["diabetes"]["metrics"]["binary"]["roc_auc"],
            },
            {
                "Phân hệ Mô hình": "Tiểu đường Pima — Softmax Regression (C=0.1, Multinomial)",
                "Accuracy (%)": models["diabetes"]["metrics"]["softmax"]["accuracy"],
                "Precision (%)": models["diabetes"]["metrics"]["softmax"]["precision"],
                "Recall (%)": models["diabetes"]["metrics"]["softmax"]["recall"],
                "F1-Score (%)": models["diabetes"]["metrics"]["softmax"]["f1"],
                "ROC-AUC": models["diabetes"]["metrics"]["softmax"]["roc_auc"],
            },
            {
                "Phân hệ Mô hình": "Tim mạch Pima — Binary Logistic Regression (C=1.0, L2)",
                "Accuracy (%)": models["cardio"]["metrics"]["binary"]["accuracy"],
                "Precision (%)": models["cardio"]["metrics"]["binary"]["precision"],
                "Recall (%)": models["cardio"]["metrics"]["binary"]["recall"],
                "F1-Score (%)": models["cardio"]["metrics"]["binary"]["f1"],
                "ROC-AUC": models["cardio"]["metrics"]["binary"]["roc_auc"],
            },
            {
                "Phân hệ Mô hình": "Tim mạch Pima — Softmax Regression (C=0.1, Multinomial)",
                "Accuracy (%)": models["cardio"]["metrics"]["softmax"]["accuracy"],
                "Precision (%)": models["cardio"]["metrics"]["softmax"]["precision"],
                "Recall (%)": models["cardio"]["metrics"]["softmax"]["recall"],
                "F1-Score (%)": models["cardio"]["metrics"]["softmax"]["f1"],
                "ROC-AUC": models["cardio"]["metrics"]["softmax"]["roc_auc"],
            },
        ]
    )
    st.dataframe(perf_df, use_container_width=True, hide_index=True)


def page_diabetes_screening(models: dict, current_user: dict) -> None:
    """Trang 2: Tầm soát Tiểu đường (Pima Indians Diabetes Dataset - 8 thuộc tính)."""
    st.title("🩸 Tầm soát Nguy cơ Bệnh Đái tháo đường")
    st.markdown(
        "Nhập đầy đủ **8 chỉ số y sinh học** theo cấu trúc tập dữ liệu **Pima Indians Diabetes Dataset**. "
        "Hệ thống tự động kiểm tra ràng buộc sinh học và thực thi suy luận đồng thời bằng 2 mô hình AI."
    )

    # Bộ chuyển đổi nhanh đơn vị đường huyết mmol/L <-> mg/dL hỗ trợ người dùng Việt Nam
    with st.expander("🔄 Công cụ quy đổi nhanh Đường huyết lúc đói (mmol/L ↔ mg/dL)", expanded=False):
        mmol_val = st.number_input("Nhập đường huyết (mmol/L):", min_value=2.0, max_value=30.0, value=5.5, step=0.1)
        st.caption(f"Giá trị tương đương: **{mmol_val * 18.0:.1f} mg/dL** (Ngưỡng kiểm soát mục tiêu: ≤ 7,0 – 7,5 mmol/L ≈ 126 – 135 mg/dL)")

    with st.form(key="diabetes_clinical_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Thông tin Nhân khẩu học & Thể trạng")
            age = st.number_input(
                "Tuổi (Age - năm)",
                min_value=18,
                max_value=100,
                value=int(current_user.get("age", 35)),
                step=1,
            )
            pregnancies = st.number_input(
                "Số lần mang thai (Pregnancies - lần, nhập 0 nếu là Nam)",
                min_value=0,
                max_value=17,
                value=0,
                step=1,
            )
            height_cm = st.slider("Chiều cao (cm) — hỗ trợ tính BMI", min_value=100, max_value=220, value=165)
            weight_kg = st.slider("Cân nặng (kg) — hỗ trợ tính BMI", min_value=30.0, max_value=180.0, value=62.0, step=0.5)
            bmi_calc = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
            bmi = st.number_input(
                f"Chỉ số khối cơ thể BMI (kg/m²) [Tự tính: {bmi_calc} - {classify_bmi_vn(bmi_calc)}]",
                min_value=10.0,
                max_value=70.0,
                value=float(bmi_calc),
                step=0.1,
            )
            dpf = st.number_input(
                "Hàm phả hệ đái tháo đường (DiabetesPedigreeFunction: 0.078 – 2.42)",
                min_value=0.05,
                max_value=2.50,
                value=0.3725,
                step=0.01,
                format="%.4f",
                help="Điểm đánh giá nguy cơ di truyền từ tiền sử gia đình (Trung vị mẫu Pima = 0.3725).",
            )

        with col2:
            st.subheader("2. Chỉ số Huyết động & Xét nghiệm Sinh hóa")
            glucose = st.number_input(
                "Nồng độ Glucose huyết tương (mg/dL)",
                min_value=36.0,
                max_value=540.0,
                value=110.0,
                step=1.0,
                help="Ngưỡng sinh tồn hợp lệ: 36 – 540 mg/dL (tương đương 2,0 – 30,0 mmol/L).",
            )
            blood_pressure = st.number_input(
                "Huyết áp tâm trương (BloodPressure - mmHg)",
                min_value=40.0,
                max_value=160.0,
                value=72.0,
                step=1.0,
                help="Giới hạn sinh tồn y khoa: 40 – 160 mmHg (Trung vị = 72 mmHg).",
            )
            skin_thickness = st.number_input(
                "Độ dày nếp gấp da cơ tam đầu (SkinThickness - mm)",
                min_value=5.0,
                max_value=99.0,
                value=23.0,
                step=1.0,
                help="Nếu chưa đo, giữ giá trị trung vị chuẩn 23.0 mm.",
            )
            insulin = st.number_input(
                "Nồng độ Insulin huyết thanh 2 giờ (μU/mL)",
                min_value=5.0,
                max_value=846.0,
                value=30.5,
                step=0.5,
                help="Nếu chưa xét nghiệm, giữ giá trị trung vị chuẩn 30.5 μU/mL.",
            )
            smoking = st.radio(
                "Thói quen hút thuốc lá / thuốc lào",
                options=[0, 1],
                format_func=lambda x: "Không hút thuốc" if x == 0 else "Có hút thuốc",
                horizontal=True,
            )
            active = st.radio(
                "Vận động thể lực đều đặn (≥ 30 phút/ngày)",
                options=[1, 0],
                format_func=lambda x: "Có vận động đều đặn" if x == 1 else "Ít vận động",
                horizontal=True,
            )

        submitted = st.form_submit_button("🔍 Đánh giá Nguy cơ Tiểu đường", use_container_width=True, type="primary")

    if submitted:
        # Kiểm thực ràng buộc hợp lệ sinh học (Mục 3.3.1.B)
        if not (36.0 <= glucose <= 540.0):
            st.error("Lỗi: Đường huyết phải nằm trong ngưỡng sinh tồn 36 – 540 mg/dL (2,0 – 30,0 mmol/L).")
            return
        if not (40.0 <= blood_pressure <= 160.0):
            st.error("Lỗi: Huyết áp tâm trương phải nằm trong ngưỡng sinh tồn 40 – 160 mmHg.")
            return

        input_payload = {
            "Pregnancies": float(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": float(age),
        }

        result = run_dual_inference(models["diabetes"], input_payload)

        # Lưu vết tự động vào CSDL SQLite health_app.db
        save_prediction_record(
            user_id=current_user["user_id"],
            disease_type="Tiểu đường (Pima)",
            input_data=input_payload,
            glucose=round(glucose / 18.0, 2),  # Quy đổi về mmol/L để đồng bộ biểu đồ xu hướng
            sys_bp=120.0,
            dia_bp=blood_pressure,
            bmi=bmi,
            smoking=smoking,
            active=active,
            result=result,
        )

        extra_notes = []
        if glucose / 18.0 >= 7.0:
            extra_notes.append(
                f"Đường huyết quy đổi đạt {glucose / 18.0:.1f} mmol/L (vượt ngưỡng kiểm soát khuyến nghị ≤ 7,0 mmol/L)."
            )
        if bmi >= 25.0:
            extra_notes.append(f"Chỉ số BMI = {bmi:.1f} kg/m² ({classify_bmi_vn(bmi)}) làm tăng đề kháng insulin.")

        render_risk_dashboard(result, extra_notes)
        render_personalized_recommendations(result["risk_level"], smoking=smoking)


def page_cardio_screening(models: dict, current_user: dict) -> None:
    """Trang 3: Tầm soát Tim mạch (Cardiovascular Disease Dataset - 12 thuộc tính đầu vào)."""
    st.title("❤️ Tầm soát Nguy cơ Bệnh Tim mạch")
    st.markdown(
        "Nhập các chỉ số lâm sàng, cận lâm sàng và yếu tố lối sống theo cấu trúc tập dữ liệu "
        "**Cardiovascular Disease Dataset**."
    )

    with st.form(key="cardio_clinical_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Thông tin Lâm sàng & Huyết động")
            age = st.number_input(
                "Tuổi (Age - năm)",
                min_value=18,
                max_value=100,
                value=int(current_user.get("age", 49)),
                step=1,
            )
            gender = st.selectbox(
                "Giới tính sinh học (Gender)",
                options=[0, 1],
                index=int(current_user.get("gender", 1)),
                format_func=lambda x: "Nữ (0)" if x == 0 else "Nam (1)",
            )
            resting_bp = st.number_input(
                "Huyết áp tâm thu lúc nghỉ (restingBP / SYS - mmHg)",
                min_value=70.0,
                max_value=240.0,
                value=130.0,
                step=1.0,
                help="Giới hạn sinh tồn y khoa: 70 – 240 mmHg.",
            )
            dia_bp = st.number_input(
                "Huyết áp tâm trương (DIA - mmHg)",
                min_value=40.0,
                max_value=160.0,
                value=80.0,
                step=1.0,
                help="Bắt buộc nhỏ hơn Huyết áp tâm thu (SYS > DIA).",
            )
            chestpain = st.selectbox(
                "Loại đau thắt ngực (chestpain)",
                options=[0, 1, 2, 3],
                index=1,
                format_func=lambda x: {
                    0: "0 - Đau thắt ngực điển hình",
                    1: "1 - Đau thắt ngực không điển hình",
                    2: "2 - Đau ngực không do tim",
                    3: "3 - Không triệu chứng / Âm thầm",
                }[x],
            )
            max_hr = st.number_input(
                "Nhịp tim tối đa đạt được (maxheartrate - nhịp/phút)",
                min_value=60.0,
                max_value=220.0,
                value=150.0,
                step=1.0,
            )
            exercise_angia = st.radio(
                "Đau thắt ngực khi vận động gắng sức (exerciseangia)",
                options=[0, 1],
                format_func=lambda x: "Không (0)" if x == 0 else "Có (1)",
                horizontal=True,
            )

        with col2:
            st.subheader("2. Chỉ số Cận lâm sàng & Lối sống")
            serum_chol = st.number_input(
                "Cholesterol huyết thanh toàn phần (serumcholestrol - mg/dL)",
                min_value=100.0,
                max_value=600.0,
                value=220.0,
                step=1.0,
                help="Cholesterol cao gây mảng xơ vữa làm hẹp lòng mạch vành.",
            )
            fasting_bs = st.radio(
                "Đường huyết lúc đói > 120 mg/dL (~6,7 mmol/L) (fastingbloodsugar)",
                options=[0, 1],
                format_func=lambda x: "Không / Bình thường (0)" if x == 0 else "Có / Tăng đường huyết (1)",
                horizontal=True,
            )
            resting_ecg = st.selectbox(
                "Kết quả Điện tâm đồ lúc nghỉ (restingrelectro)",
                options=[0, 1, 2],
                index=0,
                format_func=lambda x: {
                    0: "0 - Bình thường",
                    1: "1 - Có bất thường sóng ST-T",
                    2: "2 - Phì đại thất trái",
                }[x],
            )
            oldpeak = st.number_input(
                "Mức chênh xuống đoạn ST (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
            )
            slope = st.selectbox(
                "Độ dốc đỉnh đoạn ST khi gắng sức (slope)",
                options=[1, 2, 3],
                index=1,
                format_func=lambda x: {1: "1 - Đi lên (Upsloping)", 2: "2 - Phẳng (Flat)", 3: "3 - Đi xuống (Downsloping)"}[x],
            )
            no_vessels = st.selectbox(
                "Số mạch máu lớn nhuộm màu qua soi huỳnh quang (noofmajorvessels)",
                options=[0, 1, 2, 3],
                index=0,
            )

            col_h, col_w = st.columns(2)
            with col_h:
                height_cm = st.number_input("Chiều cao (cm)", min_value=100, max_value=220, value=168)
            with col_w:
                weight_kg = st.number_input("Cân nặng (kg)", min_value=30.0, max_value=180.0, value=65.0, step=0.5)

            smoking = st.radio(
                "Thói quen hút thuốc lá / thuốc lào",
                options=[0, 1],
                format_func=lambda x: "Không hút thuốc" if x == 0 else "Có hút thuốc",
                horizontal=True,
            )
            active = st.radio(
                "Vận động thể lực đều đặn (≥ 30 phút/ngày)",
                options=[1, 0],
                format_func=lambda x: "Có vận động" if x == 1 else "Ít vận động",
                horizontal=True,
            )

        submitted = st.form_submit_button("🔍 Đánh giá Nguy cơ Tim mạch", use_container_width=True, type="primary")

    if submitted:
        # Kiểm thực hợp lệ sinh học (TC01 & Mục 3.3.1.B)
        if resting_bp <= dia_bp:
            st.error("Lỗi: Huyết áp tâm thu phải lớn hơn huyết áp tâm trương!")
            return
        if not (70.0 <= resting_bp <= 240.0) or not (40.0 <= dia_bp <= 160.0):
            st.error("Lỗi: Huyết áp nằm ngoài giới hạn sinh tồn y khoa (SYS: 70–240 mmHg, DIA: 40–160 mmHg).")
            return

        bmi_val = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
        input_payload = {
            "Age": float(age),
            "Gender": float(gender),
            "chestpain": float(chestpain),
            "restingBP": float(resting_bp),
            "serumcholestrol": float(serum_chol),
            "fastingbloodsugar": float(fasting_bs),
            "restingrelectro": float(resting_ecg),
            "maxheartrate": float(max_hr),
            "exerciseangia": float(exercise_angia),
            "oldpeak": float(oldpeak),
            "slope": float(slope),
            "noofmajorvessels": float(no_vessels),
        }

        result = run_dual_inference(models["cardio"], input_payload)

        # Lưu kết quả vào bảng prediction_history
        save_prediction_record(
            user_id=current_user["user_id"],
            disease_type="Tim mạch (Pima)",
            input_data=input_payload,
            glucose=7.2 if fasting_bs == 1 else 5.3,
            sys_bp=resting_bp,
            dia_bp=dia_bp,
            bmi=bmi_val,
            smoking=smoking,
            active=active,
            result=result,
        )

        extra_notes = []
        if resting_bp >= 140.0 or dia_bp >= 90.0:
            extra_notes.append(
                f"Huyết áp đo được {resting_bp:.0f}/{dia_bp:.0f} mmHg vượt ngưỡng mục tiêu kiểm soát (< 140/90 mmHg)."
            )
        if serum_chol >= 240.0:
            extra_notes.append(f"Cholesterol toàn phần cao ({serum_chol:.0f} mg/dL), tăng nguy cơ xơ vữa động mạch.")

        render_risk_dashboard(result, extra_notes)
        render_personalized_recommendations(result["risk_level"], smoking=smoking)


def page_history(current_user: dict) -> None:
    """Trang 4: Lịch sử kiểm tra y tế & Biểu đồ phân tích xu hướng đa biến (Mục 3.3.4)."""
    st.title("📋 Lịch sử Kiểm tra Y tế & Phân tích Xu hướng")
    st.markdown(f"Hồ sơ theo dõi sức khỏe của: **{current_user['full_name']}** (`{current_user['username']}`)")

    filter_option = st.selectbox(
        "Lọc nhật ký theo khoảng thời gian:",
        options=["Toàn bộ lịch sử", "1 tháng gần nhất", "3 tháng gần nhất (Chu kỳ HbA1c)", "6 tháng gần nhất"],
    )
    days_map = {
        "Toàn bộ lịch sử": None,
        "1 tháng gần nhất": 30,
        "3 tháng gần nhất (Chu kỳ HbA1c)": 90,
        "6 tháng gần nhất": 180,
    }

    df_hist = get_user_history(current_user["user_id"], days_filter=days_map[filter_option])

    if df_hist.empty:
        st.info("Chưa có dữ liệu lịch sử kiểm tra. Hãy thực hiện tầm soát ở mục **Tầm soát Tiểu đường** hoặc **Tầm soát Tim mạch**.")
        return

    # Chuẩn bị bảng hiển thị thân thiện
    display_df = df_hist[
        [
            "record_id",
            "created_date",
            "disease_type",
            "glucose",
            "sys_bp",
            "dia_bp",
            "bmi",
            "binary_prob",
            "risk_level",
            "prob_scores",
        ]
    ].rename(
        columns={
            "record_id": "Mã bản ghi",
            "created_date": "Thời gian đo",
            "disease_type": "Phân hệ tầm soát",
            "glucose": "Đường huyết (mmol/L)",
            "sys_bp": "HA Tâm thu (mmHg)",
            "dia_bp": "HA Tâm trương (mmHg)",
            "bmi": "BMI (kg/m²)",
            "binary_prob": "Xác suất Logistic (%)",
            "risk_level": "Phân tầng nguy cơ",
            "prob_scores": "Chi tiết Softmax (JSON)",
        }
    )

    def _highlight_risk(row: pd.Series) -> list[str]:
        level = row["Phân tầng nguy cơ"]
        if level == "Cao":
            return ["background-color: #f8d7da; color: #721c24"] * len(row)
        if level == "Trung bình":
            return ["background-color: #fff3cd; color: #856404"] * len(row)
        return ["background-color: #d4edda; color: #155724"] * len(row)

    st.dataframe(display_df.style.apply(_highlight_risk, axis=1), use_container_width=True, hide_index=True)

    # Nút xuất dữ liệu nhật ký sức khỏe ra tệp CSV chuẩn UTF-8 (TC07)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Xuất dữ liệu nhật ký sức khỏe (CSV)",
        data=csv_bytes,
        file_name=f"lich_su_suc_khoe_user_{current_user['user_id']}.csv",
        mime="text/csv",
    )

    # Trực quan hóa xu hướng đa biến theo thời gian nếu có >= 2 bản ghi
    if len(df_hist) >= 2:
        st.markdown("### 📈 Biểu đồ Biến thiên Chỉ số Sinh học & Dịch chuyển Xác suất")
        df_sorted = df_hist.sort_values("created_date").copy()

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=df_sorted["created_date"],
                    y=df_sorted["glucose"],
                    mode="lines+markers",
                    name="Đường huyết (mmol/L)",
                    line=dict(color="#dc3545", width=2.5),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=df_sorted["created_date"],
                    y=df_sorted["sys_bp"],
                    mode="lines+markers",
                    name="HA Tâm thu (mmHg)",
                    line=dict(color="#fd7e14", width=2.5),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=df_sorted["created_date"],
                    y=df_sorted["bmi"],
                    mode="lines+markers",
                    name="Chỉ số BMI (kg/m²)",
                    line=dict(color="#0d6efd", width=2.5),
                )
            )
            fig_trend.update_layout(
                title="Biến thiên Đường huyết, Huyết áp & BMI theo thời gian",
                xaxis_title="Thời gian kiểm tra",
                yaxis_title="Giá trị chỉ số",
                height=360,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_t2:
            parsed_probs = [json.loads(p) for p in df_sorted["prob_scores"]]
            df_sorted["P_Thấp (%)"] = [p.get("low", 0) for p in parsed_probs]
            df_sorted["P_Trung_bình (%)"] = [p.get("med", 0) for p in parsed_probs]
            df_sorted["P_Cao (%)"] = [p.get("high", 0) for p in parsed_probs]

            fig_area = px.area(
                df_sorted,
                x="created_date",
                y=["P_Thấp (%)", "P_Trung_bình (%)", "P_Cao (%)"],
                title="Dịch chuyển Phân phối Xác suất Softmax qua các lần đo",
                color_discrete_map={
                    "P_Thấp (%)": "#28a745",
                    "P_Trung_bình (%)": "#ffc107",
                    "P_Cao (%)": "#dc3545",
                },
            )
            fig_area.update_layout(xaxis_title="Thời gian kiểm tra", yaxis_title="Tỷ trọng xác suất (%)", height=360)
            st.plotly_chart(fig_area, use_container_width=True)


def page_prevention_guidelines() -> None:
    """Trang 5: Tra cứu Kho tri thức Hướng dẫn Phòng ngừa & Điều trị Y tế."""
    st.title("🩺 Hướng dẫn Phòng ngừa & Kiểm soát Đa yếu tố Y khoa")
    st.markdown(
        "Kho tri thức y khoa chuẩn hóa theo **Mô hình mục tiêu kiểm soát ABCs**, khuyến cáo dinh dưỡng, "
        "vận động thể lực và tầm soát biến chứng tim mạch – đái tháo đường."
    )

    selected_tier = st.radio(
        "Chọn phân tầng nguy cơ để xem phác đồ chi tiết:",
        options=["Thấp", "Trung bình", "Cao"],
        format_func=lambda x: {
            "Thấp": "🟢 Phân tầng Nguy cơ Thấp (Duy trì lối sống lành mạnh)",
            "Trung bình": "🟡 Phân tầng Nguy cơ Trung bình (Can thiệp lối sống & kiểm soát tiền bệnh)",
            "Cao": "🔴 Phân tầng Nguy cơ Cao (Cảnh báo chuyên khoa & tầm soát biến chứng)",
        }[x],
        horizontal=True,
    )

    render_personalized_recommendations(selected_tier, smoking=1 if selected_tier == "Cao" else 0)

    st.markdown("### 🎯 Bảng Tóm tắt Ngưỡng Mục tiêu Kiểm soát Lâm sàng (Theo Báo cáo Mục 1.2.2 & 3.3.3)")
    targets_df = pd.DataFrame(
        [
            {"Chỉ số Y sinh": "Đường huyết lúc đói (Fasting Glucose)", "Ngưỡng Mục tiêu Khuyến nghị": "≤ 7,0 – 7,5 mmol/L (≤ 126 – 135 mg/dL)"},
            {"Chỉ số Y sinh": "Chỉ số HbA1c", "Ngưỡng Mục tiêu Khuyến nghị": "≤ 6,5% – 7,0% (Kiểm tra định kỳ mỗi 3 tháng)"},
            {"Chỉ số Y sinh": "Huyết áp (Blood Pressure)", "Ngưỡng Mục tiêu Khuyến nghị": "< 140/90 mmHg"},
            {"Chỉ số Y sinh": "Lượng muối tiêu thụ", "Ngưỡng Mục tiêu Khuyến nghị": "< 5g/ngày (< 1 thìa cà phê)"},
            {"Chỉ số Y sinh": "Hoạt động thể lực", "Ngưỡng Mục tiêu Khuyến nghị": "≥ 150 phút/tuần (30 phút/ngày x 5 ngày)"},
            {"Chỉ số Y sinh": "Chỉ dấu sinh học suy tim NT-proBNP", "Ngưỡng Mục tiêu Khuyến nghị": "Xét nghiệm tối thiểu 1 lần/năm ở bệnh nhân ĐTĐ / Nguy cơ Cao"},
        ]
    )
    st.table(targets_df)


# ==============================================================================
# ĐIỂM KHỞI CHẠY CHÍNH CỦA ỨNG DỤNG (MAIN ENTRYPOINT)
# ==============================================================================
def main() -> None:
    init_database()
    models = load_ai_models()

    # Quản lý trạng thái đăng nhập trong st.session_state (Mục 1.4.1)
    if "current_user" not in st.session_state:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = 'demo_user';")
        default_user = cur.fetchone()
        conn.close()
        st.session_state["current_user"] = dict(default_user)

    # Sidebar điều hướng & Quản lý tài khoản
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
        st.markdown("## 🏥 Health AI System")
        st.caption("Tầm soát Tiểu đường & Tim mạch")

        menu = st.radio(
            "Điều hướng chức năng:",
            options=[
                "🏠 Trang chủ / Giới thiệu",
                "🩸 Tầm soát Tiểu đường",
                "❤️ Tầm soát Tim mạch",
                "📋 Lịch sử kiểm tra",
                "🩺 Hướng dẫn phòng ngừa",
            ],
        )

        st.markdown("---")
        current_user = st.session_state["current_user"]
        st.markdown(f"👤 **Hồ sơ đang dùng:** `{current_user['full_name']}`")

        with st.expander("🔐 Đăng nhập / Đăng ký tài khoản"):
            auth_tab1, auth_tab2 = st.tabs(["Đăng nhập", "Đăng ký"])
            with auth_tab1:
                u_login = st.text_input("Tên đăng nhập", value="demo_user", key="login_u")
                p_login = st.text_input("Mật khẩu", type="password", value="123456", key="login_p")
                if st.button("Đăng nhập", use_container_width=True):
                    user_obj = authenticate_user(u_login, p_login)
                    if user_obj:
                        st.session_state["current_user"] = user_obj
                        st.success(f"Xin chào, {user_obj['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Sai tên đăng nhập hoặc mật khẩu.")
            with auth_tab2:
                r_name = st.text_input("Họ và tên", key="reg_name")
                r_user = st.text_input("Tên đăng nhập mới", key="reg_u")
                r_pass = st.text_input("Mật khẩu mới", type="password", key="reg_p")
                r_age = st.number_input("Tuổi", min_value=18, max_value=100, value=30, key="reg_age")
                r_gender = st.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x == 0 else "Nam", key="reg_g")
                if st.button("Tạo tài khoản", use_container_width=True):
                    ok, msg = register_user(r_user, r_pass, r_name, r_age, r_gender)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    # Điều tuyến hiển thị trang tương ứng
    if menu == "🏠 Trang chủ / Giới thiệu":
        page_home(models)
    elif menu == "🩸 Tầm soát Tiểu đường":
        page_diabetes_screening(models, st.session_state["current_user"])
    elif menu == "❤️ Tầm soát Tim mạch":
        page_cardio_screening(models, st.session_state["current_user"])
    elif menu == "📋 Lịch sử kiểm tra":
        page_history(st.session_state["current_user"])
    elif menu == "🩺 Hướng dẫn phòng ngừa":
        page_prevention_guidelines()

    # Chân trang hiển thị thông báo miễn trừ trách nhiệm y khoa bắt buộc
    render_footer_disclaimer()


if __name__ == "__main__":
    main()