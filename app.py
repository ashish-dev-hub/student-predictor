import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Main background */
.stApp {
    background: #0d0f14;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1e2230;
}

/* Cards */
.metric-card {
    background: #13161e;
    border: 1px solid #1e2230;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #4f8ef7; }

.score-display {
    font-family: 'Syne', sans-serif;
    font-size: 72px;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.grade-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 16px;
    margin-top: 8px;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f8ef7;
    margin-bottom: 12px;
}

/* Slider labels */
.stSlider label { color: #9ca3af !important; font-size: 13px !important; }

/* Select boxes */
.stSelectbox label { color: #9ca3af !important; font-size: 13px !important; }

/* Divider */
hr { border-color: #1e2230 !important; }

/* Plotly charts bg */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background: #1e2230 !important;
    color: #e8e6e0 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Info boxes */
.tip-box {
    background: #0f1520;
    border: 1px solid #1a2540;
    border-left: 3px solid #4f8ef7;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #8ba4d4;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load('ridge_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    with open('feature_names.json') as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv('student_habits_performance.csv')

try:
    model, scaler, feature_names = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model files nahi mile: {e}\n\nPehle Colab mein Cell 18 chalao aur `.pkl` files is folder mein rakho.")


# ─── Sidebar — Input Form ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.markdown("---")

    st.markdown('<div class="section-header">📚 Study Habits</div>', unsafe_allow_html=True)
    study_hours        = st.slider("Study hours per day", 0.0, 9.0, 4.0, 0.1)
    attendance         = st.slider("Attendance (%)", 56.0, 100.0, 85.0, 0.5)

    st.markdown("---")
    st.markdown('<div class="section-header">📱 Screen Time</div>', unsafe_allow_html=True)
    social_media       = st.slider("Social media hours/day", 0.0, 8.0, 2.5, 0.1)
    netflix            = st.slider("Netflix/streaming hours/day", 0.0, 6.0, 1.8, 0.1)

    st.markdown("---")
    st.markdown('<div class="section-header">🏃 Lifestyle</div>', unsafe_allow_html=True)
    sleep_hours        = st.slider("Sleep hours/day", 3.0, 10.0, 6.5, 0.1)
    exercise           = st.slider("Exercise days/week", 0, 6, 3)
    mental_health      = st.slider("Mental health rating (1–10)", 1, 10, 6)

    st.markdown("---")
    st.markdown('<div class="section-header">👤 Background</div>', unsafe_allow_html=True)
    age                = st.slider("Age", 17, 24, 20)
    gender             = st.selectbox("Gender", ["Female", "Male", "Other"])
    diet_quality       = st.selectbox("Diet quality", ["Poor", "Fair", "Good"])
    internet_quality   = st.selectbox("Internet quality", ["Poor", "Average", "Good"])
    parental_edu       = st.selectbox("Parent education", ["High School", "Bachelor", "Master"])
    part_time_job      = st.selectbox("Part-time job?", ["No", "Yes"])
    extracurricular    = st.selectbox("Extracurricular?", ["No", "Yes"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Score")


# ─── Encode Input ─────────────────────────────────────────────
def encode_input():
    diet_map     = {"Poor": 0, "Fair": 1, "Good": 2}
    inet_map     = {"Poor": 0, "Average": 1, "Good": 2}
    edu_map      = {"High School": 0, "Bachelor": 1, "Master": 2}
    gender_map   = {"Female": 0, "Male": 1, "Other": 2}

    row = {
        "age":                          age,
        "gender":                       gender_map[gender],
        "study_hours_per_day":          study_hours,
        "social_media_hours":           social_media,
        "netflix_hours":                netflix,
        "part_time_job":                1 if part_time_job == "Yes" else 0,
        "attendance_percentage":        attendance,
        "sleep_hours":                  sleep_hours,
        "diet_quality":                 diet_map[diet_quality],
        "exercise_frequency":           exercise,
        "parental_education_level":     edu_map[parental_edu],
        "internet_quality":             inet_map[internet_quality],
        "mental_health_rating":         mental_health,
        "extracurricular_participation":1 if extracurricular == "Yes" else 0,
    }
    return pd.DataFrame([row])[feature_names]

def get_grade(score):
    if score >= 90: return "A+", "#22c55e", "Outstanding"
    if score >= 80: return "A",  "#4ade80", "Excellent"
    if score >= 70: return "B",  "#4f8ef7", "Good"
    if score >= 60: return "C",  "#f59e0b", "Average"
    if score >= 50: return "D",  "#f97316", "Below Average"
    return "F", "#ef4444", "Needs Improvement"

def get_tips(score, study_hours, attendance, sleep_hours, social_media, mental_health):
    tips = []
    if study_hours < 3:
        tips.append("📚 Increase your study hours — aim for at least 3–4 hours per day")
    if attendance < 80:
        tips.append("🏫 Maintain attendance above 80% — it has a direct impact on your score")
    if sleep_hours < 6:
        tips.append("😴 Get 7–8 hours of sleep — it is essential for brain consolidation")
    if social_media > 4:
        tips.append("📵 Reduce social media usage to less than 2 hours — it’s a major source of distraction")
    if mental_health < 5:
        tips.append("🧘 Pay attention to your mental health — try counseling or meditation")
    if not tips:
        tips.append(" All habits are good! Maintain consistency")
    return tips


# ─── Main Content ─────────────────────────────────────────────
st.markdown("# STUDENT PERFORMANCE PRDEICTOR")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA & Insights", "ℹ️ About Model"])

# ══ TAB 1: PREDICTION ══════════════════════════════════════════
with tab1:
    if model_loaded and predict_btn:
        input_df   = encode_input()
        prediction = float(model.predict(input_df)[0])
        prediction = max(0, min(100, prediction))
        grade, color, label = get_grade(prediction)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#9ca3af;font-size:13px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px">Predicted Exam Score</div>
                <div class="score-display">{prediction:.1f}</div>
                <div class="grade-badge" style="background:{color}22;color:{color};border:1px solid {color}55">
                    Grade {grade} · {label}
                </div>
                <div style="color:#6b7280;font-size:12px;margin-top:16px">out of 100</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#4a5568"},
                    'bar': {'color': "#4f8ef7", 'thickness': 0.25},
                    'bgcolor': "#1e2230",
                    'bordercolor': "#2d3748",
                    'steps': [
                        {'range': [0,  50], 'color': '#1a1010'},
                        {'range': [50, 70], 'color': '#1a1505'},
                        {'range': [70, 85], 'color': '#0a1520'},
                        {'range': [85,100], 'color': '#0a1a10'},
                    ],
                    'threshold': {'line': {'color': color, 'width': 3}, 'value': prediction}
                },
                number={'font': {'color': '#e8e6e0', 'size': 36}, 'suffix': '/100'}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#13161e', plot_bgcolor='#13161e',
                font_color='#9ca3af', height=220, margin=dict(t=20, b=10, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Habit analysis radar
            habits = {
                'Study Hours':    min(study_hours / 8 * 100, 100),
                'Attendance':     attendance,
                'Sleep Quality':  min(sleep_hours / 9 * 100, 100),
                'Mental Health':  mental_health * 10,
                'Exercise':       exercise / 6 * 100,
                'Diet':           {"Poor":20,"Fair":55,"Good":90}[diet_quality],
            }
            categories = list(habits.keys())
            values     = list(habits.values())
            values    += [values[0]]
            categories += [categories[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=values, theta=categories,
                fill='toself',
                fillcolor='rgba(79,142,247,0.15)',
                line=dict(color='#4f8ef7', width=2),
                marker=dict(color='#4f8ef7', size=6)
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#13161e',
                    radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color='#6b7280',size=10), gridcolor='#1e2230', linecolor='#1e2230'),
                    angularaxis=dict(tickfont=dict(color='#9ca3af',size=12), gridcolor='#1e2230', linecolor='#1e2230')
                ),
                paper_bgcolor='#13161e', plot_bgcolor='#13161e',
                showlegend=False, height=300,
                margin=dict(t=30, b=30, l=60, r=60)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Tips section
        st.markdown("### 💡 Personalized Tips")
        tips = get_tips(prediction, study_hours, attendance, sleep_hours, social_media, mental_health)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

    elif model_loaded:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#4a5568">
            <div style="font-size:48px;margin-bottom:16px">👈</div>
            <div style="font-family:'Syne',sans-serif;font-size:20px;color:#6b7280">
                Fill the student details in the sidebar<br>And click on <strong style="color:#4f8ef7">Predict Score</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══ TAB 2: EDA ═════════════════════════════════════════════════
with tab2:
    if model_loaded:
        DARK = dict(paper_bgcolor='#13161e', plot_bgcolor='#13161e',
                    font_color='#9ca3af', margin=dict(t=40,b=20,l=20,r=20))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Exam Score Distribution**")
            fig1 = px.histogram(df, x='exam_score', nbins=30,
                                color_discrete_sequence=['#4f8ef7'],
                                labels={'exam_score':'Exam Score'})
            fig1.update_layout(**DARK, height=300)
            fig1.update_traces(marker_line_width=0, opacity=0.8)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**Study Hours vs Exam Score**")
            fig2 = px.scatter(df, x='study_hours_per_day', y='exam_score',
                              color='diet_quality',
                              color_discrete_map={'Poor':'#ef4444','Fair':'#f59e0b','Good':'#22c55e'},
                              labels={'study_hours_per_day':'Study Hrs/Day','exam_score':'Exam Score'})
            fig2.update_layout(**DARK, height=300)
            fig2.update_traces(marker=dict(size=5, opacity=0.7))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Attendance vs Exam Score**")
            fig3 = px.scatter(df, x='attendance_percentage', y='exam_score',
                              color='internet_quality',
                              color_discrete_map={'Poor':'#ef4444','Average':'#f59e0b','Good':'#22c55e'},
                              labels={'attendance_percentage':'Attendance %','exam_score':'Exam Score'})
            fig3.update_layout(**DARK, height=300)
            fig3.update_traces(marker=dict(size=5, opacity=0.7))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown("**Avg Score by Diet Quality**")
            diet_avg = df.groupby('diet_quality')['exam_score'].mean().reset_index()
            fig4 = px.bar(diet_avg, x='diet_quality', y='exam_score',
                          color='diet_quality',
                          color_discrete_map={'Poor':'#ef4444','Fair':'#f59e0b','Good':'#22c55e'},
                          labels={'diet_quality':'Diet Quality','exam_score':'Avg Exam Score'})
            fig4.update_layout(**DARK, height=300, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        # Correlation heatmap
        st.markdown("**Correlation Heatmap (Numeric Features)**")
        num_cols = df.select_dtypes(include=np.number).drop(columns=['age']).corr()
        fig5 = px.imshow(num_cols, color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, aspect='auto',
                         text_auto='.2f')
        fig5.update_layout(**DARK, height=420)
        fig5.update_traces(textfont=dict(size=10))
        st.plotly_chart(fig5, use_container_width=True)


# ══ TAB 3: ABOUT MODEL ═════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 Model Details")
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | **Algorithm** | Ridge Regression |
        | **Target** | `exam_score` (0–100) |
        | **Features** | 14 student habits |
        | **Train/Test** | 80% / 20% |
        | **Tuning** | RandomizedSearchCV (5-fold CV) |
        | **Best R²** | ~0.83–0.87 |
        """)

    with col2:
        st.markdown("### 📏 Evaluation Metrics")
        st.markdown("""
        | Metric | Description |
        |---|---|
        | **R²** | Variance explained (higher = better) |
        | **RMSE** | Penalizes large errors more |
        | **MAE** | Average absolute prediction error |
        """)

    st.markdown("### 🔢 Ridge Coefficients (Feature Impact)")
    st.markdown("*🟢 Positive = score badhta hai    🔴 Negative = score girta hai*")

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)

    fig_feat = px.bar(coef_df, x='Coefficient', y='Feature',
                      orientation='h',
                      labels={'Coefficient': 'Coefficient Value', 'Feature': ''},
                      color='Coefficient',
                      color_continuous_scale=['#ef4444', '#1e2230', '#22c55e'],
                      range_color=[-coef_df['Coefficient'].abs().max(),
                                    coef_df['Coefficient'].abs().max()])
    fig_feat.add_vline(x=0, line_dash='dash', line_color='#6b7280', line_width=1)
    fig_feat.update_layout(
        paper_bgcolor='#13161e', plot_bgcolor='#13161e',
        font_color='#9ca3af', height=420,
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.markdown(" BUILT FOR BDCOE ")
