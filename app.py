import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import random

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Support Ticket Classification",
    page_icon="🎫",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("category_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("all_tickets_processed_improved_v3.csv")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp{
    background-color:#F8FAFC;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #E5E7EB;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🎫 Ticket Classifier")

st.sidebar.markdown("""
### Features

✅ Ticket Classification

✅ NLP Processing

✅ TF-IDF Vectorization

✅ LinearSVC Model

✅ 85% Accuracy

✅ Analytics Dashboard
""")

# =========================
# TITLE
# =========================

st.title("🎫 Support Ticket Classification & Prioritization")

st.markdown("""
Automatically classify customer support tickets using
Natural Language Processing (NLP) and Machine Learning.
""")

# =========================
# METRICS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "📨 Total Tickets",
        len(df)
    )

with col2:
    st.metric(
        "🎯 Model Accuracy",
        "85.18%"
    )

# =========================
# CATEGORY DISTRIBUTION
# =========================

st.subheader("📊 Ticket Category Distribution")

category_counts = df["Topic_group"].value_counts()

fig, ax = plt.subplots(figsize=(8,4))

category_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Category")
ax.set_ylabel("Count")

st.pyplot(fig)

# =========================
# LIVE PREDICTION
# =========================

st.subheader("🤖 Live Ticket Analyzer")

ticket_id = st.text_input("🎫 Enter Ticket ID")
customer_name = st.text_input("👤 Customer Name")

department = st.selectbox(
    "🏢 Department",
    ["HR", "Finance", "IT", "Operations", "Sales", "Marketing"]
)
ticket_text = st.text_area(
    "Enter Ticket Description"
)
team_mapping = {
    "Access": "IAM Team 🔐",
    "Administrative rights": "Admin Support Team 👨‍💼",
    "Hardware": "Hardware Support Team 💻",
    "Network": "Network Team 🌐",
    "Software": "Application Support Team 🖥️",
    "Email": "Messaging Team 📧",
    "Security": "Security Operations Team 🛡️",
    "Database": "Database Team 🗄️",
    "Infrastructure": "Infrastructure Team 🏢",
    "Applications": "Application Team ⚙️",
    "Miscellaneous": "General Support Team 📞"
}
if st.button("Predict Category"):

    if ticket_text.strip() != "":

        text_vector = vectorizer.transform([ticket_text])

        prediction = model.predict(text_vector)[0]

        generated_ticket_id = f"TKT-{random.randint(1000,9999)}"

        assigned_team = team_mapping.get(
            prediction,
            "General Support Team 📞"
        )

        high_keywords = [
            "server",
            "down",
            "urgent",
            "critical",
            "security",
            "breach",
            "failure"
        ]

        medium_keywords = [
            "slow",
            "error",
            "issue",
            "problem"
        ]

        text_lower = ticket_text.lower()

        if any(word in text_lower for word in high_keywords):
            priority = "HIGH 🔴"

        elif any(word in text_lower for word in medium_keywords):
            priority = "MEDIUM 🟡"

        else:
            priority = "LOW 🟢"

        st.info(f"🎫 Ticket ID: {ticket_id}")
        st.info(f"👤 Customer: {customer_name}")
        st.info(f"🏢 Department: {department}")

        st.success(f"Predicted Category: {prediction}")

        st.info(f"Assigned Team: {assigned_team}")

        st.warning(f"Priority Level: {priority}")

        if prediction.lower() == "access":
            resolution = """
- Verify user credentials
- Reset password if required
- Check account permissions
"""

        elif prediction.lower() == "hardware":
            resolution = """
- Check hardware connections
- Restart affected device
- Escalate to hardware support team
"""

        else:
            resolution = """
- Review ticket details
- Assign to appropriate support team
"""

        st.info(f"Suggested Resolution:\n\n{resolution}")

# =========================
# DATA PREVIEW
# =========================
st.subheader("📌 Business Insights")

st.info("""
• Hardware-related tickets are the most common support requests.

• Access-related issues represent a significant portion of tickets.

• Automated classification reduces manual ticket routing effort.

• NLP-based ticket categorization improves response efficiency.

• Machine Learning enables faster support operations.
""")