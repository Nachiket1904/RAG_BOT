import streamlit as st
import pdfplumber
import openai
from gtts import gTTS
import os
import tempfile
import io

# Set up OpenAI API key
openai.api_key = "OPEN-AI-KEY"

# Load the PDF file
with open("Nachi_resume_ML.pdf", "rb") as pdf_file:
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
    # Add information about the college and the app
    context = (
        f"Context: {pdf_text}\n\n"
        "This app was developed by an MSc AI student at BK Birla College, a prestigious educational institution in Kalyan, India. "
        "The app is designed to answer questions related to today's NAAC (National Assessment and Accreditation Council) presentation at the college."
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
import tempfile
from gtts import gTTS
from gtts.lang import tts_langs

def text_to_speech(text, language='en'):
    """
    Converts the given text to speech and returns the audio data.

    Args:
        text (str): The text to be converted to speech.
        language (str, optional): The language code. Default is 'en'.

    Returns:
        bytes: The audio data in MP3 format.
    """
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        # Check if the language is supported
        if language not in tts_langs().keys():
            raise ValueError(f"Language '{language}' is not supported.")

        # Get the first available voice for the specified language
        available_voices = tts_langs().get(language)
        if not available_voices:
            raise ValueError(f"No voices are available for language '{language}'.")
        voice = available_voices[0]

        tts = gTTS(text=text, lang=language, tld='com', slow=False)
        tts.save(fp.name + ".mp3")
        audio_file = open(fp.name + ".mp3", "rb")
        audio_bytes = audio_file.read()
    return audio_bytes

# Streamlit app
def main():
    st.title("Voice-Based NAAC Presentation Chatbot")
    st.write("This chatbot can answer questions related to today's NAAC presentation at BK Birla College.")

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_data)

    # Get user question
    user_question = st.text_input("Ask a question about the NAAC presentation:")

    if user_question:
        # Generate answer
        answer = get_answer(user_question, pdf_text)
        st.write(f"Answer: {answer}")

        # Option to convert answer to audio
        if st.button("Convert to Audio"):
            audio_bytes = text_to_speech(answer)
            st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()
