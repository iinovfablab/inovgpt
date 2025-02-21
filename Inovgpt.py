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


text = ""

prev_prompt = """Voce será um especialista no assunto Inovfablab, impressoras 3d e cortadora a laser, 
            você sera armazenara as explicações toda vez que a palavra 'seci aprender' for chamada, você irá aprender, apenas com esta palavra.
            Você ira restringir perguntas com assunto de politica, futebol, matemática, programação
            e tudo que nao for relacionado a inovfablab ou a impressoras 3d e cortadora a laser, 
            será respondido educadamente como 'Infelizmente não fui programada para responder ou ajudar com esses tipo de pergunta'.

            Cortadora laser: A cortadora laser, é uma CNC de 2 dimensões, utiliza um laser de CO2(gas carbonico) para fazer a remoção do material,
            ela possui um sistema de espelhos que possibilita a reflexao dos laser até o bico principal, possui um compressor de ar
            que faz a remoção e evita a entrada de residuos no bico.
            A cortadora a laser possui um gabarito com medidas utilizadas para determina a altura do bico, dessa forma ajusta o foco do laser,
            a altura utilizada no momento é de 5.5mm, o primeiro degrau, onde o aluno irá ajustar o bico, soltando o parafuso dourado,
            no sentido anti-horario

            Impressoras 3D: O inovfablab possui 5 impressoras 3D, 3 finders, 1 Guider 2s e uma bambo x1-carbon, todas FDM.
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
        speed=1
    )

    for i, (gs, ps, audio) in enumerate(generator):

        sf.write(f'{audio_path}'+'\\'+f'{i}.wav', audio, 24000) # save each audio file
    #path = os.listdir(os.path.abspath('.'))
    #audio_file = list(filter(lambda x: x.endswith(".wav"), path))

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



async def chaT(context):
    path = os.listdir(os.path.abspath('.'))
    audio_file = list(filter(lambda x: x.endswith(".wav"), path))


    chat = ""
    print("chat")
    while not chat.endswith("exit"):

        c = random.choice([30,31,32,33,34,35,36,37,90,91,92,93,94,95,96,97])
        b = random.choice([40,41,42,43,45,46,47,100,101,102,103,104,105,106,107])
        text=""
        chat = input("\nHumano>>: ")
        for part in ollama.generate(model="llama3.2:3B", prompt=chat, context=context,stream=True):    
            #print(f'\033[92m'+f'\033[42m'+part['response']+'\033[97m', end='', flush=True)
            print(part['response'], end='', flush=True)
            text+=part['response']

        

        await generate_audio(text)
        await play()
        remove_wav_files()

        #subprocess.Popen(["python", "sound.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        

async def main(context):
    task1 = asyncio.create_task(chaT(context))

    await asyncio.gather(task1)


    
    

if __name__ == "__main__":
    

    context = prev_generate(prev_prompt)
    asyncio.run(main(context))
    #playsound("0.wav")
    
