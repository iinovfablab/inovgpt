import io
import os
import json
import wave
import wave
import torch
import shutil
import asyncio
import pyaudio  
import datetime
import requests
import contextlib
from kokoro import KPipeline
import speech_recognition as sr
from langchain.llms import Ollama
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, initialize_agent, AgentType




KEYWORDS = ["ceci", "seci", "sessi", "cessi"]

template_ = """
Você é um assistente que responde apenas com **um nome de máquina** da lista abaixo, com base na pergunta do usuário.

## Regras:
- Apenas retorne o nome completo de uma máquina da lista.
- Não explique nada.
- Não adicione comentários.
- Não diga "não sei".
- Mesmo com nomes incompletos ou errados, retorne o mais próximo.
- NUNCA escreva nada além do nome exato da máquina.
- Retorne só 1 linha com o nome.

## Lista de máquinas:
- IMPRESSORA 3D FINDER - 2
- Prototipadora de circuitos impressos (PCI)
- Parafusadeira e Furadeira a Bateria
- Furadeira de Bancada
- Lixadeira portátil DEWALT
- Serra tico-tico Bosch
- Maquina de costura SINGER
- Estação de solda e retrabalho
- Micro Retífica
- IMPRESSORA 3D FINDER - 3
- IMPRESSORA 3D ZMorph
- Gravadora de Matrize a vácuo
- Impressora 3D - Guider IIs
- IMPRESSORA 3D FINDER - 1
- Cortadora e Gravadora a Laser
- Plotter de recorte
- X1 Carbon Combo – Impressora 3D
- Bambu Lab A1 - (1)
- Bambu Lab A1 - (2)

## Exemplos:
Pergunta: Quem está na próxima reserva da laser?
Resposta: Cortadora e Gravadora a Laser

Pergunta: Qual o nome correto da finder 2?
Resposta: IMPRESSORA 3D FINDER - 2

Pergunta: Quem usa a guider 2s no próximo horário?
Resposta: Impressora 3D - Guider IIs

Pergunta: {question}
Resposta:
"""

template__= """Você é um assistente que responde quem está utilizando a {machine_name} no proximo horário.

            Base de dados de reservas possui inicio e fim:
            {base_json}

            Agora são {real_time}.

            Pergunta: {question}

            Responda de forma curta e objetiva.
            """
            
template___ = """Responda perguntas frequentes dos usuários sobre o laboratório Inovfablab e suas máquinas, resumidamente, sem fugir do contexto da pergunta.
                
                
                CONTEXTO1:   
                            Cortadora laser: A cortadora laser, é uma CNC de 2 dimensões, utiliza um laser de CO2(gas carbonico) para fazer a remoção do material,
                            ela possui um sistema de espelhos que possibilita a reflexao dos laser até o bico principal, possui um compressor de ar
                            que faz a remoção e evita a entrada de residuos no bico.
                            A cortadora a laser possui um gabarito com medidas utilizadas para determina a altura do bico, dessa forma ajusta o foco do laser,
                            a altura utilizada no momento é de 5.5mm, o primeiro degrau, onde o aluno irá ajustar o bico, soltando o parafuso dourado,
                            no sentido anti-horario

                      
                CONTEXTO2: 
                            Impressoras 3D: A impressão 3D é uma forma de tecnologia de fabricação aditiva onde um modelo tridimensional é criado por sucessivas camadas de material a partir de um modelo 3D digital.
                             Por não necessitar do uso de moldes e permitir produzir formas que não são viáveis ou práticas de se conseguir em outros métodos de produção, tem vantagens em relação a outras tecnologias de fabricação mecânica,
                            como por exemplo a injeção de plástico, desbaste ou modelagem/modelação, sendo mais rápida e mais barata em diversos casos. Essas tecnicas, 
                            oferecem aos desenvolvedores de produtos a habilidade de num simples processo imprimirem partes de alguns materiais com diferentes propriedades físicas e mecânicas. 
                            Alguns modelos de impressoras industriais podem utilizar uma boa variedade de materiais como polímeros sintéticos e orgânicos, 
                            concreto, metais e até mesmo alimentos, possuindo milhares de cores, permitindo criar protótipos com boa precisão, aparência e funcionalidades dos produtos.

                            Fabricação aditiva é o processo de criar objetos a partir de modelos digitais criados em três dimensões. As tecnologias de fabricação aditiva compreendem a fusão a laser, 
                            fundição a vácuo e moldagem por injeção.A fusão a laser é um processo de fabricação aditiva digital que utiliza energia laser concentrada para fundir pós metálicos em objetos 3D. 
                            A fusão a laser é uma tecnologia de fabricação emergente, com presença na indústria médica (ortopedia), aeroespacial, assim como nos diversos setores da engenharia e dos serviços.

                            A fundição a vácuo basicamente é utilizada para produzir protótipos de alta qualidade em variedade de resinas de poliuretano (PU) que mimetizam o desenho de polímeros de engenharia. 
                            O nylon também pode ser fundido a vácuo e criar matrizes de cera para processos de fundição de cera perdida. Algumas máquinas injetoras são apropriadas para a produção de pequenas séries, 
                            utilizando molde de resina, ou produção em série de pequenas peças pesando até 12 gramas dependendo do modelo escolhido.

                CONTEXTO3:  
                            Um plotter de recorte é uma máquina que usa uma lâmina para cortar materiais finos e flexíveis, como vinil adesivo, papel, tecido, entre outros, seguindo desenhos criados em um software no computador. Diferente de uma impressora comum, que imprime imagens, o plotter de recorte corta as formas desejadas, sendo ideal para criar adesivos, etiquetas, decalques, transfers e outros produtos personalizados.
                            

                CONTEXTO4:
                            Uma prototipadora de circuito impresso, ou prototipadora PCB, é uma máquina que permite a fabricação de placas de circuito impresso (PCB) em pequena escala, essencialmente para fins de prototipagem. Elas são usadas para criar protótipos de circuitos eletrônicos, facilitando a validação de projetos e a criação de versões de teste.

                CONTEXTO5:
                            O laboratório InovFabLab está sediado na Universidade Santa Cecília, no térreo, ao lado do banco Itaú. Ele se encontra na Rua Oswaldo Cruz, 277 – Boqueirão – Santos/SP. CEP: 11045-907 | Tel.: (13) 3202-7100 / (13) 2104-7150.
                            
                
                Pergunta: {question}
                Resposta
            """

prefix = """Vocé é um agent supervisor que deve escolher a ferramenta correta de acordo com a pergunta.
            Você possui as seguintes ferramentas:"""


suffix = """Você é um agente inteligente com acesso a ferramentas externas. Sua única função é **redirecionar a pergunta para a ferramenta correta**, sem modificar ou interpretar a pergunta.

            Você deve:
            - Sempre encaminhar a pergunta original como está, dentro de Action Input.

            Ferramentas disponíveis:
            - JsonTool: para perguntas sobre horários, reservas, consultar a base de dados.
            - FablabTool: para perguntas sobre o laboratório, funcionamento, ou dúvidas sobre máquinas.

            Question: {input}
            """

def real_time(_):
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

def data_json(mach_name):
    data = json.loads(requests.get(f"http://127.0.0.1:5000/{mach_name.strip()}").content)
    return data

def wrapper_json(query):
    return chainManager()[0].invoke({"question": query})

def wrapper_fablab(query):
    return chainManager()[1].invoke({"question":query})

def toolManager():
    json_tool = Tool(
    "JsonTool",
    func=wrapper_json,
    description="Use esta ferramenta para perguntas sobre reservas, horários.",
    return_direct=True
    )

    fablab_expert_tool = Tool(
        "FablabTool",
        func=wrapper_fablab,
        description="Use esta ferramenta para responder perguntas sobre o funcionamento do InovFabLab, impressoras 3D ou cortadora a laser.",
        return_direct=True
    )

    tools = [json_tool, fablab_expert_tool]
    return tools

def chainManager():
    llm_ = Ollama(model="mistral:7B")
    prompt_mname = ChatPromptTemplate.from_template(template_)
    prompt_assist = ChatPromptTemplate.from_template(template__)
    prompt_fablab = ChatPromptTemplate.from_template(template___)

    expert_fab_chain = (RunnablePassthrough()
                        |prompt_fablab
                        |llm_
                        |StrOutputParser())

    
    name_chain = (
        RunnablePassthrough()
        | prompt_mname
        | llm_
        | StrOutputParser()
        )

    assist_chain = (
        RunnablePassthrough().assign(machine_name=name_chain).assign(real_time=real_time,
                                                                    base_json=lambda x: data_json(x["machine_name"]))
        | prompt_assist
        | llm_
        | StrOutputParser()
        )

    return [assist_chain, expert_fab_chain]

def agentManager():
    
    llm_ = Ollama(model="gemma2:27b")
    
    custom_prompt = ZeroShotAgent.create_prompt(
    tools=toolManager(),
    prefix=prefix,
    suffix=suffix,
    input_variables=["input"]

    )

    memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        return_messages=True
    )
    agent = initialize_agent(
    memory=memory,
    tools=toolManager(),
    llm=llm_,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs={"prompt":custom_prompt},
    return_imediate_steps=True
    )
    return agent

async def generate_audio(text):
    
    local_path = os.path.abspath('.')

    #if not os.path.isdir("audio_"):
    #    os.mkdir("audio_")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipeline = KPipeline(lang_code='p',device=device)
    generator = pipeline(
        text, voice='pf_dora',
        speed=1
    )
  

    for i, (_, _, audio) in enumerate(generator):
        #sf.write(f'{audio_path}'+'\\'+f'{i}.wav', audio, 24000)
        
        audio_int16 = (audio * 44200).to(torch.int16)  
        buff = io.BytesIO()
        torch.save(audio_int16, buff)
        buff.seek(0)
        p = pyaudio.PyAudio()  

        #f = wave.open(buff, "rb")

        stream = p.open(format = pyaudio.paInt16,  
                        channels = 1,  
                        rate = 24000,  
                        output = True) 


        
        stream.write(buff.read())

        stream.stop_stream()
        stream.close()
    p.terminate()            

def speech_text(phase):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        
    try:
        if phase == 1:
            while True:
                text = recognizer.recognize_google(audio, language="pt-BR").lower()
                print(text)
                for word in KEYWORDS:

                    if word in text:
                        return "True"

                    else:
                        return "False"
        if phase == 2:

            text = recognizer.recognize_google(audio, language="pt-BR")
            return text
    except:
        return "nada"
    

async def chaT(chain):

    chat = ""

    while not chat.endswith("sair"):

        text = ""
        try:
            if speech_text(1) == "True":
                print('\a')
                print("estou ouvindo...")
                chat = speech_text(2)
                print(chat)
                
            else:
                continue
        except:
            continue

        # for part in chain.stream({"question":chat}):    
        #     print(part, end='', flush=True)
        #     text+=part
        
        text = chain.invoke({"input":chat})["output"]    
             

        
        text.replace("/", "de")
        await generate_audio(text)
        
async def main(chain):
    task1 = asyncio.create_task(chaT(chain))

    await asyncio.gather(task1)


if __name__ == "__main__":
    #context = prev_generate(prev_prompt)
    agent = agentManager()
    chain = chainManager()
    asyncio.run(main(agent))