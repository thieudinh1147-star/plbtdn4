"""
HỆ THỐNG ĐÁNH GIÁ NGUY CƠ & CHẨN ĐOÁN SỨC KHỎE
Ứng dụng Streamlit dùng mô hình THẬT đã huấn luyện (scaler, pca, kmeans,
random_forest). File random_forest.joblib quá nặng để đưa lên GitHub nên
được host trên Google Drive và tự động tải về khi app khởi động.
"""

import json
import os

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================================
# CẤU HÌNH ĐƯỜNG DẪN MODEL
# =========================================================================
MODEL_DIR = "model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

# File ID lấy từ link chia sẻ Google Drive của random_forest.joblib
# (đoạn nằm giữa /d/ và /view trong link chia sẻ)
RF_DRIVE_FILE_ID = "1t6qrndkCO2B9QvVxBnv7JZ-h0r4l9JFc"
RF_LOCAL_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")

# Ánh xạ giá trị tiếng Việt trên giao diện -> giá trị gốc lúc train model
GENDER_MAP = {"Nam": "Male", "Nữ": "Female", "Khác": "Other"}
SMOKING_MAP = {
    "Không rõ": "No Info",
    "Chưa từng hút": "never",
    "Đã từng hút, bỏ lâu": "former",
    "Từng hút (ever)": "ever",
    "Mới bỏ gần đây": "not current",
    "Đang hút": "current",
}

# Phải khớp CHÍNH XÁC với danh sách cột đã áp StandardScaler lúc train
# trong notebook (chỉ 4 cột số liên tục, không gồm cột nhị phân/one-hot)
NUMERIC_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

# =========================================================================
# CẤU HÌNH TRANG & CUSTOM CSS (Phong cách Medical UI)
# =========================================================================
st.set_page_config(
    page_title="Đánh giá nguy cơ sức khỏe",
    page_icon=":stethoscope:",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* Nền tổng thể xám trắng nhạt, dịu mắt */
    .stApp {
        background-color: #F8F9FA;
    }

    /* Khối tiêu đề (Header) */
    .header-box {
        background: linear-gradient(135deg, #E8F6F3 0%, #E3F2FD 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    .header-box h1 {
        color: #1B4F4C;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .header-box p {
        color: #4A5568;
        font-size: 15px;
        margin: 0;
    }

    /* Thẻ chứa chung (form, kết quả, biểu đồ) */
    .card-box {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #EAECEE;
    }
    .card-title {
        color: #1B4F4C;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 16px;
    }

    /* Nút submit nổi bật */
    div.stFormSubmitButton > button {
        background-color: #17A589;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 10px 0;
        width: 100%;
        font-size: 15px;
        transition: background-color 0.2s ease-in-out;
    }
    div.stFormSubmitButton > button:hover {
        background-color: #148F77;
        color: white;
    }

    /* Ô kết quả chẩn đoán / nhóm nguy cơ */
    .result-label {
        font-size: 26px;
        font-weight: 700;
        margin: 4px 0 12px 0;
    }
    .result-sub {
        font-size: 13px;
        color: #718096;
        margin-bottom: 6px;
    }
    .badge-safe   { color: #1E8449; background-color: #EAFAF1; }
    .badge-medium { color: #B9770E; background-color: #FEF5E7; }
    .badge-danger { color: #C0392B; background-color: #FDEDEC; }
    .badge-critical { color: #FFFFFF; background-color: #8E1B1B; }

    .badge-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 15px;
    }

    .recommend-box {
        background-color: #F4F9F9;
        border-left: 4px solid #17A589;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
    }
    .recommend-box ul {
        margin: 0;
        padding-left: 18px;
    }
    .recommend-box li {
        margin-bottom: 6px;
        color: #333333;
        font-size: 14px;
    }

    /* Ép nhãn (label) của mọi ô nhập liệu luôn hiển thị màu tối, tránh bị
       ẩn mất khi trình duyệt người dùng đang bật chế độ tối (dark mode) -
       lúc đó Streamlit tự đổi label sang màu trắng, trùng với nền sáng
       mà CSS này đang ép, khiến chữ biến mất. */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    .stApp label {
        color: #2D3748 !important;
        font-weight: 500 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# KHỐI 1: TIÊU ĐỀ (HEADER)
# =========================================================================
st.markdown(
    """
    <div class="header-box">
        <h1>HỆ THỐNG ĐÁNH GIÁ NGUY CƠ & CHẨN ĐOÁN SỨC KHỎE</h1>
        <p>Nhập thông tin sức khỏe của bệnh nhân để nhận kết quả chẩn đoán sơ bộ
        và phân khúc nhóm nguy cơ tiểu đường, kèm theo khuyến nghị chăm sóc phù hợp.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# LOAD MODEL THẬT (chạy 1 lần, cache lại nhờ st.cache_resource)
# =========================================================================
@st.cache_resource
def load_models():
    """Load scaler/pca/kmeans từ thư mục model_artifacts/ (đã có trong repo
    GitHub). Riêng random_forest.joblib quá nặng nên tải về từ Google Drive
    trước, chỉ tải 1 lần rồi dùng lại cho các lần dự đoán sau."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))

    if not os.path.exists(RF_LOCAL_PATH):
        gdown.download(id=RF_DRIVE_FILE_ID, output=RF_LOCAL_PATH, quiet=False)
    rf_model = joblib.load(RF_LOCAL_PATH)

    with open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return scaler, pca, kmeans, rf_model, meta


scaler, pca, kmeans, rf_model, meta = load_models()
FEATURE_COLUMNS = meta["feature_columns"]
CLUSTER_TO_RISK = {int(k): v for k, v in meta["cluster_to_risk"].items()}
RECOMMENDATION_LOOKUP = meta["recommendation_lookup"]

# Tâm của từng cụm K-Means (dữ liệu THẬT lấy từ model đã train, không bịa).
# Chỉ lấy 2 chiều đầu (PC1, PC2) trong không gian PCA 9 chiều để vẽ lên
# biểu đồ 2D - đây là cách đơn giản hóa để minh họa, không phải suy diễn.
CLUSTER_CENTERS_DF = pd.DataFrame(
    kmeans.cluster_centers_[:, :2], columns=["PC1", "PC2"]
)
CLUSTER_CENTERS_DF["Nhóm nguy cơ"] = [
    CLUSTER_TO_RISK.get(i, f"Cụm {i}") for i in range(len(CLUSTER_CENTERS_DF))
]


def _encode_raw_input(raw: dict) -> pd.DataFrame:
    """Mã hóa dữ liệu thô giống hệt bước tiền xử lý lúc train.

    Lưu ý: không dùng pd.get_dummies() ở đây vì nó chỉ hoạt động đúng khi
    thấy đủ các giá trị khác nhau trong dữ liệu; với 1 dòng duy nhất (1
    bệnh nhân), get_dummies() sẽ luôn tự loại bỏ cột one-hot do chỉ thấy
    1 giá trị -> kết quả luôn sai (mọi lựa chọn trên form đều bị hiểu
    thành nhóm mặc định). Thay vào đó, gán thẳng giá trị 1 cho đúng cột
    one-hot tương ứng, dựa theo danh sách FEATURE_COLUMNS đã lưu lúc train.
    """
    encoded = {col: 0 for col in FEATURE_COLUMNS}

    for col in [
        "age", "hypertension", "heart_disease",
        "bmi", "HbA1c_level", "blood_glucose_level",
    ]:
        if col in encoded:
            encoded[col] = raw[col]

    gender_col = f"gender_{raw['gender']}"
    if gender_col in encoded:
        encoded[gender_col] = 1

    smoking_col = f"smoking_history_{raw['smoking_history']}"
    if smoking_col in encoded:
        encoded[smoking_col] = 1

    return pd.DataFrame([encoded])[FEATURE_COLUMNS]


def _scale_input(X_new: pd.DataFrame) -> np.ndarray:
    """Chỉ chuẩn hóa 4 cột số liên tục bằng scaler đã lưu, giữ nguyên các
    cột nhị phân/one-hot - phải khớp đúng cách xử lý lúc train (StandardScaler
    KHÔNG được áp cho các cột phân loại, tránh phóng đại nhóm hiếm gặp)."""
    other_cols = [c for c in X_new.columns if c not in NUMERIC_COLS]
    X_num_scaled = scaler.transform(X_new[NUMERIC_COLS])
    X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=NUMERIC_COLS, index=X_new.index)
    X_scaled_df = pd.concat([X_num_scaled_df, X_new[other_cols]], axis=1)[X_new.columns]
    return X_scaled_df.values


def predict_pipeline(inputs: dict) -> dict:
    """Nhận dict thông tin bệnh nhân (giá trị tiếng Việt từ form), trả về
    dict kết quả từ 2 luồng của mô hình thật:
    - label, proba (Luồng 2 - Random Forest)
    - risk_group, recommendations (Luồng 1 - PCA + K-Means)
    - pca_point (tọa độ thật trên 2 thành phần đầu của không gian PCA)
    """
    raw = {
        "age": inputs["age"],
        "bmi": inputs["bmi"],
        "HbA1c_level": inputs["HbA1c"],
        "blood_glucose_level": inputs["glucose"],
        "hypertension": 1 if inputs["hypertension"] == "Có" else 0,
        "heart_disease": 1 if inputs["heart_disease"] == "Có" else 0,
        "gender": GENDER_MAP[inputs["gender"]],
        "smoking_history": SMOKING_MAP[inputs["smoking_history"]],
    }

    X_new = _encode_raw_input(raw)
    X_new_scaled = _scale_input(X_new)

    # ---- Luồng 1: PCA -> K-Means (phân khúc rủi ro) ----
    X_new_pca = pca.transform(X_new_scaled)
    cluster_id = int(kmeans.predict(X_new_pca)[0])
    risk_group = CLUSTER_TO_RISK.get(cluster_id, "Không xác định")
    recommendations = RECOMMENDATION_LOOKUP.get(risk_group, [])
    if isinstance(recommendations, str):
        recommendations = [recommendations]

    # ---- Luồng 2: Random Forest (chẩn đoán chính xác) ----
    proba = float(rf_model.predict_proba(X_new_scaled)[0, 1])
    label = "BỆNH" if rf_model.predict(X_new_scaled)[0] == 1 else "KHÔNG BỆNH"

    # Chỉ lấy 2 thành phần đầu của PCA (9D) để vẽ lên biểu đồ 2D
    pc1, pc2 = float(X_new_pca[0, 0]), float(X_new_pca[0, 1])

    return {
        "label": label,
        "proba": proba,
        "risk_group": risk_group,
        "recommendations": recommendations,
        "pca_point": (pc1, pc2),
    }


# =========================================================================
# KHỐI 2: FORM NHẬP LIỆU
# =========================================================================
st.markdown('<div class="card-box">', unsafe_allow_html=True)
st.markdown('<p class="card-title">Thông tin bệnh nhân</p>', unsafe_allow_html=True)

with st.form(key="patient_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])

    with col2:
        bmi = st.number_input(
            "Chỉ số BMI", min_value=10.0, max_value=50.0, value=24.5, step=0.1
        )
        smoking_history = st.selectbox(
            "Tiền sử hút thuốc", list(SMOKING_MAP.keys())
        )

    with col3:
        hba1c = st.number_input(
            "Chỉ số HbA1c", min_value=3.5, max_value=9.0, value=5.5, step=0.1
        )
        hypertension = st.selectbox("Tăng huyết áp", ["Không", "Có"])

    with col4:
        glucose = st.number_input(
            "Mức đường huyết", min_value=70, max_value=300, value=100
        )
        heart_disease = st.selectbox("Bệnh tim", ["Không", "Có"])

    submitted = st.form_submit_button(label="DỰ ĐOÁN KẾT QUẢ")

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# XỬ LÝ SAU KHI NHẤN NÚT DỰ ĐOÁN
# =========================================================================
if submitted:
    inputs = {
        "age": age,
        "bmi": bmi,
        "HbA1c": hba1c,
        "glucose": glucose,
        "gender": gender,
        "smoking_history": smoking_history,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
    }
    st.session_state["last_result"] = predict_pipeline(inputs)

result = st.session_state.get("last_result")

# =========================================================================
# KHỐI 3: KẾT QUẢ DỰ ĐOÁN (2 CỘT)
# =========================================================================
if result is not None:
    col_left, col_right = st.columns(2)

    # ---- CỘT TRÁI: Luồng 2 - Dự đoán bệnh ----
    with col_left:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-title">Kết quả chẩn đoán</p>', unsafe_allow_html=True
        )

        proba_pct = round(result["proba"] * 100, 1)
        is_danger = result["proba"] > 0.5
        label_color = "#C0392B" if is_danger else "#1E8449"
        bar_color = "#E74C3C" if is_danger else "#2ECC71"
        bg_color = "#FDEDEC" if is_danger else "#EAFAF1"

        st.markdown(
            f"""
            <p class="result-label" style="color: {label_color};">{result['label']}</p>
            <p class="result-sub">Xác suất mắc bệnh</p>
            <div style="background-color: {bg_color}; border-radius: 999px; height: 14px; overflow: hidden;">
                <div style="width: {proba_pct}%; background-color: {bar_color}; height: 100%;"></div>
            </div>
            <p style="color: {label_color}; font-weight: 600; margin-top: 8px;">{proba_pct}%</p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- CỘT PHẢI: Luồng 1 - Nhóm nguy cơ & khuyến nghị ----
    with col_right:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-title">Phân khúc nguy cơ</p>', unsafe_allow_html=True
        )

        badge_class = {
            "Nguy cơ Thấp": "badge-safe",
            "Nguy cơ Trung bình": "badge-medium",
            "Nguy cơ Cao": "badge-danger",
            "Nguy cơ Rất cao": "badge-critical",
        }.get(result["risk_group"], "badge-medium")

        recommendations_html = "".join(
            f"<li>{item}</li>" for item in result["recommendations"]
        )

        st.markdown(
            f"""
            <span class="badge-pill {badge_class}">{result['risk_group']}</span>
            <div class="recommend-box">
                <ul>{recommendations_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # KHỐI 4: TRỰC QUAN HÓA PCA 2D — dùng tâm cụm THẬT từ model K-Means
    # =====================================================================
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-title">Vị trí bệnh nhân so với tâm các nhóm nguy cơ (PC1-PC2)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Biểu đồ hiển thị tâm thật của 4 cụm K-Means (chỉ chiếu 2/9 chiều PCA "
        "để minh họa trên mặt phẳng) và vị trí thật của bệnh nhân hiện tại."
    )

    color_map = {
        "Nguy cơ Thấp": "#2ECC71",
        "Nguy cơ Trung bình": "#F39C12",
        "Nguy cơ Cao": "#E74C3C",
        "Nguy cơ Rất cao": "#8E1B1B",
    }

    fig = px.scatter(
        CLUSTER_CENTERS_DF,
        x="PC1",
        y="PC2",
        color="Nhóm nguy cơ",
        color_discrete_map=color_map,
        text="Nhóm nguy cơ",
    )
    fig.update_traces(marker=dict(size=22, symbol="diamond"), textposition="top center")

    patient_x, patient_y = result["pca_point"]
    fig.add_trace(
        go.Scatter(
            x=[patient_x],
            y=[patient_y],
            mode="markers",
            marker=dict(
                symbol="star",
                size=22,
                color="#1B4F4C",
                line=dict(color="white", width=1.5),
            ),
            name="Bệnh nhân hiện tại",
        )
    )

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Vui lòng nhập thông tin bệnh nhân và bấm 'DỰ ĐOÁN KẾT QUẢ' để xem kết quả.")

# =========================================================================
# CẢNH BÁO: KẾT QUẢ CHỈ MANG TÍNH THAM KHẢO (luôn hiển thị, không phụ
# thuộc việc đã bấm dự đoán hay chưa)
# =========================================================================
st.warning(
    "**Lưu ý:** Kết quả trên chỉ mang tính chất **tham khảo**, được tạo ra "
    "bởi mô hình học máy dựa trên dữ liệu thống kê, **không phải chẩn đoán "
    "y khoa chính thức**. Vui lòng đến cơ sở y tế/bệnh viện để được bác sĩ "
    "thăm khám, xét nghiệm và kiểm chứng trước khi đưa ra bất kỳ quyết định "
    "điều trị nào."
)
