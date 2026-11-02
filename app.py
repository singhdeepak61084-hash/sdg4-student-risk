import streamlit as st, pandas as pd, joblib

st.set_page_config(page_title="At-Risk Student Predictor", page_icon="🎓")
st.title("🎓 At-Risk Student Predictor (SDG 4: Quality Education)")
st.caption("Enter a student's details to estimate the chance they score below the pass threshold, so support can be offered early.")

@st.cache_resource
def load(): return joblib.load("model.joblib")
model = load()

L3 = ["Low", "Medium", "High"]
c1, c2 = st.columns(2)
with c1:
    hours = st.slider("Hours studied per week", 1, 44, 20)
    att = st.slider("Attendance (%)", 60, 100, 85)
    sleep = st.slider("Sleep hours per night", 4, 10, 7)
    prev = st.slider("Previous score", 50, 100, 70)
    tut = st.slider("Tutoring sessions per month", 0, 8, 1)
    phys = st.slider("Physical activity (hrs/week)", 0, 6, 3)
    parent = st.selectbox("Parental involvement", L3, 1)
    access = st.selectbox("Access to resources", L3, 1)
    motiv = st.selectbox("Motivation level", L3, 1)
with c2:
    income = st.selectbox("Family income", L3, 1)
    teacher = st.selectbox("Teacher quality", L3, 1)
    extra = st.selectbox("Extracurricular activities", ["Yes", "No"])
    net = st.selectbox("Internet access", ["Yes", "No"])
    ld = st.selectbox("Learning disabilities", ["No", "Yes"])
    school = st.selectbox("School type", ["Public", "Private"])
    peer = st.selectbox("Peer influence", ["Positive", "Neutral", "Negative"])
    pedu = st.selectbox("Parental education", ["High School", "College", "Postgraduate"])
    dist = st.selectbox("Distance from home", ["Near", "Moderate", "Far"])
    gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict risk", type="primary"):
    row = pd.DataFrame([{
        "Hours_Studied": hours, "Attendance": att, "Parental_Involvement": parent,
        "Access_to_Resources": access, "Extracurricular_Activities": extra, "Sleep_Hours": sleep,
        "Previous_Scores": prev, "Motivation_Level": motiv, "Internet_Access": net,
        "Tutoring_Sessions": tut, "Family_Income": income, "Teacher_Quality": teacher,
        "School_Type": school, "Peer_Influence": peer, "Physical_Activity": phys,
        "Learning_Disabilities": ld, "Parental_Education_Level": pedu,
        "Distance_from_Home": dist, "Gender": gender}])
    p = model.predict_proba(row)[0, 1]
    st.metric("Probability of being at risk", f"{p:.0%}")
    st.progress(float(p))
    if p >= 0.5: st.error("High risk: consider tutoring, attendance follow-up and counselling.")
    else: st.success("Low risk: keep monitoring progress.")
