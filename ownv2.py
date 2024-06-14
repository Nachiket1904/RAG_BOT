import streamlit as st
import pdfplumber
import openai
from gtts import gTTS
import os
import tempfile
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set up OpenAI API key
openai.api_key = os.getenv('OPEN_AI_KEY')

# Load the PDF file
with open("NAAC.pdf", "rb") as pdf_file:
    pdf_data = pdf_file.read()

# PDF text extraction function
def extract_text_from_pdf(pdf_data):
    with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Question answering function using OpenAI API
def get_answer(question, pdf_text):
    # Add information about the college, the app, and the CS/IT department
    context = (
        f"Context: {pdf_text}\n\n"
        "This app was developed by an MSc AI student at BK Birla College, a prestigious educational institution in Kalyan, India. "
        "The app is designed to answer questions related to the Department of Computer Science and Information Technology. "
        "The CS/IT department offers undergraduate and postgraduate programs, conducts research, and organizes various technical events and workshops."
    )

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"{context}\n\nQuestion: {question}\nAnswer:",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()

# Text-to-Speech function
def text_to_speech(text, language='en'):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text, lang=language, tld='com', slow=False)
        tts.save(fp.name + ".mp3")
        audio_file = open(fp.name + ".mp3", "rb")
        audio_bytes = audio_file.read()
    return audio_bytes

# Streamlit app
def main():
    st.set_page_config(page_title="Robby - Your Voice Assistant", page_icon="🎓", layout="wide")

    st.title("🎓 Robby - Your Voice Assistant")
    st.markdown("""
    Welcome to Robby, your intelligent voice assistant for all things related to BK Birla College's  Department of Computer Science & IT!

    💡 Ask me anything about:
    - The Department of Computer Science and IT
    - Programs offered, research activities, and events

    I'm here to help you with instant answers and voice responses. Let's explore together!
    """)

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_data)

    # Get user question
    user_question = st.text_input("💬 What would you like to know?")

    if user_question:
        # Generate answer
        answer = get_answer(user_question, pdf_text)
        st.write(f"🤖 Answer: {answer}")

        # Option to convert answer to audio
        if st.button("🔊 Listen to the Answer"):
            audio_bytes = text_to_speech(answer)
            st.audio(audio_bytes, format="audio/mp3")

    st.sidebar.title("About Robby")
    st.sidebar.info(
        "Robby is an AI-powered assistant developed by an MSc AI student at BK Birla College. "
        "It's designed to provide quick and accurate information about the Department of Computer Science & IT. Enjoy a seamless experience with both text and voice responses!"
    )

if __name__ == "__main__":
    main()
