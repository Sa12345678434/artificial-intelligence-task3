import whisper
import sounddevice as sd
import numpy as np
import queue
import cohere
import pyttsx3  


model = whisper.load_model("base")


co = cohere.Client("My API key")  

engine = pyttsx3.init()


q = queue.Queue()
samplerate = 16000
duration = 5  

def audio_c(indata, frames, time, status):
    q.put(indata.copy())

def record_audio():
    print("Speak now -recording for 5 sec-...")
    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_c):
        audio = []
        for _ in range(int(samplerate / 1024 * duration)):
            audio.append(q.get())
        audio_np = np.concatenate(audio, axis=0).flatten()
    return audio_np

def transcribe_audio(audio_np):
    audio_np = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio_np).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

def ask_cohere(prompt):
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0.7
    )
    return response.text

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    audio = record_audio()
    user_text = transcribe_audio(audio)
    print("You:", user_text)

    if user_text.strip() == "":
        print("No speech detected.")
        return
    print("The response is being built...")
    reply = ask_cohere(user_text)
    print("Cohere:", reply)

    
    speak_text(reply)



if __name__ == "__main__":
    main()
