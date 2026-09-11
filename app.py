# import streamlit as st
# import pickle

# st.set_page_config(
# page_title="Email Spam Detector",
# page_icon="📧",
# layout="centered"
# )

# with open("tfidf.pkl", "rb") as file:
#     tfidf = pickle.load(file)


# with open("email_spam_model.pkl", "rb") as file:
#     model = pickle.load(file)


# st.title("📧 Email Spam Detection")

# st.write(
# "Enter an email below and the machine learning model "
# "will predict whether it is Spam or Not Spam."
# )
# st.divider()

# emai_text = st.text_area(
#     "✉️ Enter Email",
#     placeholder="Paste or type your email here...",
#     height=250
# )

# if st.button("🔍 Check Email", use_container_width=True):
#     if emai_text.strip()== "":
#         st.warning("⚠️ Please enter an email first.")
#     else:
#         email_tfidf = tfidf.transform([emai_text])
#         prediction = model.predict(email_tfidf)[0]

#         if prediction == 1:
#             st.error("🚨 This is a SPAM email.")
#         else:
#             st.success("✅ This is NOT a SPAM email.")


#         st.write("Prediction value:", prediction)
#     st.divider()

# st.caption("Email Spam Detection | TF-IDF + Multinomial Naive Bayes")





import streamlit as st
import pickle

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 50%,
        #e0f2fe 100%
    );
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #475569;
    margin-bottom: 25px;
}

.info-card {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
}

.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

with open("tfidf.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("email_spam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.markdown(
    '<div class="main-title">📧 Email Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered email classification using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">

📌 <b>How it works</b><br><br>

Enter an email message below.<br>
The TF-IDF vectorizer converts the text into numerical features.<br>
Then the Multinomial Naive Bayes model predicts the email type.

</div>
""", unsafe_allow_html=True)

emai_text = st.text_area(
    "✉️ Enter Email",
    placeholder="Paste or type the email content here...",
    height=250
)

if st.button("🔍 Check Email", use_container_width=True):

    if emai_text.strip() == "":
        st.warning("⚠️ Please enter an email first.")

    else:
        email_tfidf = tfidf.transform([emai_text])

        prediction = model.predict(email_tfidf)[0]

        if prediction == 1:

            st.error("🚨 SPAM EMAIL DETECTED")

            st.markdown("""
            <div class="result-card">

            ⚠️ <b>Warning!</b><br><br>

            This email has been classified as <b>Spam</b>.<br>
            Be careful with links, attachments and requests for personal information.

            </div>
            """, unsafe_allow_html=True)

        else:

            st.success("✅ NOT A SPAM EMAIL")

            st.markdown("""
            <div class="result-card">

            🛡️ <b>Email looks safe</b><br><br>

            This email has been classified as <b>Not Spam</b>.

            </div>
            """, unsafe_allow_html=True)

        st.write("Prediction value:", prediction)

st.divider()

st.markdown(
    '<div class="footer">'
    'Email Spam Detection | TF-IDF + Multinomial Naive Bayes'
    '</div>',
    unsafe_allow_html=True
)


