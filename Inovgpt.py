from time import time
import ollama
from kokoro import KPipeline
import soundfile as sf
import os
import subprocess
import wave
import contextlib
import asyncio
import random
import pyaudio  
import wave
import shutil
import speech_recognition as sr

KEYWORDS = ["ceci", "seci", "sessi", "cessi"]

prev_prompt = """Voce será um especialista no assunto Inovfablab, impressoras 3d e cortadora a laser, 

            Você ira restringir perguntas com assunto de politica, futebol, matemática, programação
            e tudo que não for relacionado a inovfablab ou a impressoras 3d e cortadora a laser, 
            será respondido educadamente como 'Infelizmente não fui programada para responder ou ajudar com esses tipo de pergunta'.

            Cortadora laser: A cortadora laser, é uma CNC de 2 dimensões, utiliza um laser de CO2(gas carbonico) para fazer a remoção do material,
            ela possui um sistema de espelhos que possibilita a reflexao dos laser até o bico principal, possui um compressor de ar
            que faz a remoção e evita a entrada de residuos no bico.
            A cortadora a laser possui um gabarito com medidas utilizadas para determina a altura do bico, dessa forma ajusta o foco do laser,
            a altura utilizada no momento é de 5.5mm, o primeiro degrau, onde o aluno irá ajustar o bico, soltando o parafuso dourado,
            no sentido anti-horario

            Impressoras 3D: O inovfablab possui 5 impressoras 3D, 3 finders, 1 Guider 2s e uma bambo x1-carbon, todas FDM.

            O horário de funcionamento do inovfablab é das 08:00 da manhão até as 22:30 da noite, funcionando de segunda a sexta neste horarios,
            aos sábados o horário é das 09:30h até as 12:30h, pois ninguem é de ferro né.

            existem três funcionarios responsaveis que mantém a integridade do laboratório que são Ricardo que fica no horário da manhã e tarde, José que fica no horário da manhã e noite, 
            sendo gerenciado pelo Responsavel Sergio Schina que fica a noite.

            -Entenda-se que se a pessoa falar "i9", ela esta se referindo ao INOVFABLAB.
            -Entenda-se que se a pessoa falar "inove fablab", ela esta se referindo ao INOVFABLAB
            """



def prev_generate(prompt_):
    output = ollama.generate(model="llama3.2:1B", prompt=prompt_,stream=True)
    
    for chunk in output:
        if chunk['done'] == True:
            print('First generate complete!')
            context = chunk['context']
    return context
    

async def generate_audio(text):
    
    local_path = os.path.abspath('.')
    audio_path = os.path.join(local_path, "audio_")

    if not os.path.isdir("audio_"):
        os.mkdir("audio_")
    
    
    pipeline = KPipeline(lang_code='p')
    generator = pipeline(
        text, voice='pf_dora',
        speed=1.2
    )

    for i, (_, _, audio) in enumerate(generator):
        sf.write(f'{audio_path}'+'\\'+f'{i}.wav', audio, 24000)


async def play():


    local_path = os.path.abspath('.')
    audio_path = os.path.join(local_path, "audio_")

    chunk = 1024  

    for x in os.listdir(audio_path):
        f = wave.open(audio_path+"\\"+x,"rb")  
        print(audio_path+"\\"+x)
        p = pyaudio.PyAudio()  

        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)  

        data = f.readframes(chunk)  

        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  

        stream.stop_stream()  
        stream.close()  

        p.terminate() 
    

        
def remove_wav_files():

    local_path = os.path.abspath('.')
    audio_path = os.path.join(local_path, "audio_")
    shutil.rmtree(audio_path)
        


def sound_duration(wav):
    
    with contextlib.closing(wave.open(wav,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def speech_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
    
    try:
        text = recognizer.recognize_google(audio, language="pt-BR").lower()
        print(text)
        for word in KEYWORDS:

            if word in text:
                return text

            else:
                return "nada"
    except:
        return "nada"



async def chaT(context):
    path = os.listdir(os.path.abspath('.'))
    audio_file = list(filter(lambda x: x.endswith(".wav"), path))


    chat = ""
    while not chat.endswith("sair"):

        text = ""
        print("ouvindo...")
        chat = speech_text()
        if chat.endswith("nada"):
            continue

        for part in ollama.generate(model="llama3.2:3B", prompt=chat, context=context,stream=True):    
            print(part['response'], end='', flush=True)
            text+=part['response']

        

        await generate_audio(text)
        await play()
        remove_wav_files()


async def main(context):
    task1 = asyncio.create_task(chaT(context))

    await asyncio.gather(task1)


    
    

if __name__ == "__main__":
    

    context = prev_generate(prev_prompt)
    asyncio.run(main(context))

    
