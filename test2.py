import streamlit as st
import torch
import torchaudio
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
from tortoise.utils.text import split_and_recombine_text
from time import time
import warnings
import logging

# Configurar logging para depuração
logging.basicConfig(level=logging.INFO)

# Suprimir avisos futuros para uma interface mais limpa
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração Inicial do Streamlit
st.set_page_config(page_title="Clonagem de Voz com Tortoise TTS", layout="wide")
st.title("Aplicação de Clonagem de Voz")

st.write("""
Esta aplicação permite selecionar uma voz personalizada, inserir um texto e gerar o áudio correspondente.
""")

# Definir o diretório base das vozes
VOICE_BASE_DIR = "tortoise/voices/"  # Caminho relativo ao diretório atual

# Inicialização do Tortoise TTS com cache para melhorar desempenho
@st.cache_resource
def initialize_tts():
    return TextToSpeech()

tts = initialize_tts()

# Função para carregar vozes personalizadas mapeando nomes para nomes de pastas
def load_custom_voices():
    return {
        "Voz 1": "voz1",
        "Voz 2": "voz2",
        "Voz 3": "voz3"
    }

# Carregar vozes personalizadas
voices = load_custom_voices()

# Seção de Seleção de Voz
st.header("Selecione a Voz Personalizada:")

# Inicializar as variáveis de sessão
if 'selected_voice' not in st.session_state:
    st.session_state.selected_voice = None

# Função para selecionar a voz e atualizar o estado
def select_voice(voice_name):
    st.session_state.selected_voice = voice_name
    st.success(f"Voz '{voice_name}' selecionada!")
    logging.info(f"Voz selecionada: {voice_name}")

# Exibir botões para cada voz
for voice_name in voices.keys():
    if st.button(voice_name):
        select_voice(voice_name)

# Verificar se uma voz foi selecionada
if st.session_state.selected_voice is None:
    st.warning("Por favor, selecione uma voz para continuar.")
else:
    # Input de Texto
    st.header("Insira o Texto para Conversão em Áudio:")
    text_input = st.text_area("Texto:", 
                              value="""We were good, we were gold, Kind of dream that can't be sold, We were right 'til we weren't, Built a home and watched it burn""",
                              height=150)

    # Botão para Gerar Áudio
    if st.button("Gerar Áudio"):
        with st.spinner("Gerando áudio..."):
            try:
                logging.info("Iniciando a geração do áudio.")
                
                # Preparação do Texto
                if '|' in text_input:
                    texts = text_input.split('|')
                else:
                    texts = split_and_recombine_text(text_input)
                
                seed = int(time())
                
                # Diretório para salvar os resultados
                voice_dir = st.session_state.selected_voice.replace(' ', '_')  # Nome único baseado no nome da voz
                outpath = os.path.join("results", "longform", voice_dir)
                os.makedirs(outpath, exist_ok=True)
                
                # Carregar amostras de voz usando o nome da voz
                voice_name = st.session_state.selected_voice
                voice_samples, conditioning_latents = load_voice(voice_name)
                
                # Verificar se as amostras foram carregadas corretamente
                if voice_samples is None or conditioning_latents is None:
                    raise ValueError(f"Falha ao carregar a voz '{voice_name}' a partir de: {VOICE_BASE_DIR}{voice_name}/")
                
                all_parts = []
                for j, text_part in enumerate(texts):
                    gen = tts.tts_with_preset(
                        text_part, 
                        voice_samples=voice_samples, 
                        conditioning_latents=conditioning_latents,
                        preset="fast", 
                        k=1, 
                        use_deterministic_seed=seed
                    )
                    gen = gen.squeeze(0).cpu()
                    part_path = os.path.join(outpath, f'{j}.wav')
                    torchaudio.save(part_path, gen, 24000)
                    all_parts.append(gen)
                    logging.info(f"Parte {j} do áudio gerada e salva em {part_path}.")
                
                # Concatenar todas as partes em um único áudio
                full_audio = torch.cat(all_parts, dim=-1)
                combined_path = os.path.join(outpath, 'combined.wav')
                torchaudio.save(combined_path, full_audio, 24000)
                logging.info(f"Áudio combinado salvo em {combined_path}.")
                
                # Reproduzir o áudio gerado diretamente no Streamlit
                with open(combined_path, 'rb') as f:
                    audio_bytes = f.read()
                
                st.audio(audio_bytes, format='audio/wav')
                st.success("Áudio gerado com sucesso!")
                logging.info("Áudio gerado e reproduzido com sucesso.")
            
            except Exception as e:
                logging.error(f"Erro na geração do áudio: {e}")
                st.error(f"Ocorreu um erro durante a geração do áudio: {e}")
