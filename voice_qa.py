import speech_recognition as sr
import pyttsx3

from text_generation import generate_text


def listen_question():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(audio)

        return text

    except Exception:

        return "Could not understand audio."


def answer_voice_question():

    question = listen_question()

    prompt = f"Question: {question}\nAnswer:"

    answer = generate_text(prompt)

    short_answer = answer.split(".")[0]

    engine = pyttsx3.init()

    try:

        engine.say(short_answer)

        engine.runAndWait()

    except RuntimeError:

        pass

    return question, short_answer