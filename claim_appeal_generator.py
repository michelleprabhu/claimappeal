import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import requests
import re
from datetime import datetime

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text if text else "No text extracted."

# Function to extract patient details from medical records
def extract_patient_info(medical_text):
    name = re.search(r"Patient Name:\s*(.*)", medical_text)
    address = re.search(r"Address:\s*(.*)", medical_text)
    phone = re.search(r"Phone Number:\s*(.*)", medical_text)
    email = re.search(r"Email:\s*(.*)", medical_text)

    return {
        "name": name.group(1) if name else "[Patient Name]",
        "address": address.group(1) if address else "[Patient Address]",
        "phone": phone.group(1) if phone else "[Patient Phone]",
        "email": email.group(1) if email else "[Patient Email]"
    }

# Initialize AI model using RouterLLM
def initialize_agent(api_key):
    try:
        # Call RouterLLM to determine the best model
        router_response = requests.post("http://localhost:8000/routerllm", json={"query": "Summarize medical appeal"})
        router_data = router_response.json()
        selected_model = router_data.get("model", "gpt-4")

        llm = ChatOpenAI(
            temperature=0.7,
            model=selected_model,
            openai_api_key=api_key
        )
        memory = ConversationBufferMemory()
        return ConversationChain(llm=llm, memory=memory)
    except Exception as e:
        st.error(f"Error initializing AI agent: {e}")
        return None

# Streamlit UI setup
st.set_page_config(page_title="Medical Claim Appeal Generator", page_icon="🩺", layout="wide")

# Header
st.markdown("<h1 style='margin-top: 10px;'>Medical Claim Appeal Generator</h1>", unsafe_allow_html=True)
st.write("Upload medical documents to generate a professional appeal letter.")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# File uploaders
st.header("Upload Documents")
eob_file = st.file_uploader("Upload Explanation of Benefits (EOB)", type=["pdf"])
medical_file = st.file_uploader("Upload Medical Records", type=["pdf"])
denial_file = st.file_uploader("Upload Denial Letter", type=["pdf"])

# Preview uploaded documents
if eob_file:
    st.subheader("EOB Preview:")
    eob_text = extract_text_from_pdf(eob_file)
    st.text_area("EOB Content", eob_text, height=200)

if medical_file:
    st.subheader("Medical Records Preview:")
    medical_text = extract_text_from_pdf(medical_file)
    st.text_area("Medical Records Content", medical_text, height=200)

if denial_file:
    st.subheader("Denial Letter Preview:")
    denial_text = extract_text_from_pdf(denial_file)
    st.text_area("Denial Letter Content", denial_text, height=200)

# Generate Appeal Letter
if st.button("Generate Appeal Letter"):
    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    elif eob_file and medical_file and denial_file:
        st.write("Initializing AI agent...")
        agent = initialize_agent(api_key)

        if agent is None:
            st.error("Failed to initialize AI agent.")
        else:
            patient_info = extract_patient_info(medical_text)

            appeal_prompt = f"""
            Generate an appeal letter based on:
            1. Explanation of Benefits (EOB): {eob_text}
            2. Medical Records: {medical_text}
            3. Denial Letter: {denial_text}

            The letter should:
            - Be professional and persuasive.
            - Clearly explain the reason for appeal.
            - Justify medical necessity.
            - Start with the patient's details.

            Patient Name: {patient_info['name']}
            """

            with st.spinner("Generating appeal letter..."):
                try:
                    appeal_letter = agent.run(appeal_prompt)

                    # ✅ Log query in MIP Monitoring
                    log_data = {
                        "query": "Generate medical appeal",
                        "model": agent.llm.model_name,
                        "latency": f"{round(datetime.now().timestamp() % 10, 2)}s",
                        "cost": "$0.03" if agent.llm.model_name == "gpt-4" else "$0.00"
                    }
                    requests.post("http://localhost:8000/logs", json=log_data)

                    # Display results
                    st.subheader("Generated Appeal Letter:")
                    st.text_area("Appeal Letter", appeal_letter, height=400)

                    # Download button
                    st.download_button("Download Appeal Letter", appeal_letter, "appeal_letter.txt", "text/plain")

                except Exception as e:
                    st.error(f"Error generating appeal letter: {e}")
    else:
        st.error("Please upload all required documents.")

