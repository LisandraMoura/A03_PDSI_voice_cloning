import streamlit as st
import torch
import torchaudio
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
from tortoise.utils.text import split_and_recombine_text
from time import time
import warnings

# Suprimir avisos futuros para uma interface mais limpa
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração Inicial do Streamlit
st.set_page_config(page_title="Clonagem de Voz com Tortoise TTS", layout="wide")
st.title("Aplicação de Clonagem de Voz")

st.write("""
Esta aplicação permite selecionar uma voz personalizada, inserir um texto e gerar o áudio correspondente.
""")

# Inicialização do Tortoise TTS com cache para melhorar desempenho
@st.cache_resource
def initialize_tts():
    return TextToSpeech()

tts = initialize_tts()

# Função para carregar vozes personalizadas de caminhos fornecidos
def load_custom_voices():
    return {
        "Voz 1": "/home/lisamenezes/Searches/A03_PDSI_voice_cloning/voices_cloning_app/voices/voz1/Miley_cortes1.wav",
        "Voz 2": "/home/lisamenezes/Searches/A03_PDSI_voice_cloning/voices_cloning_app/voices/voz1/Miley_cortes1.wav",
        "Voz 3": "/home/lisamenezes/Searches/A03_PDSI_voice_cloning/voices_cloning_app/voices/voz1/Miley_cortes1.wav"
    }

# Carregar vozes personalizadas
voices = load_custom_voices()

# Seleção de Voz
st.write("Selecione a voz personalizada:")
for voice_name, voice_path in voices.items():
    if st.button(voice_name):
        selected_voice = voice_path
        st.success(f"Voz '{voice_name}' selecionada!")

# # Verificar se uma voz foi selecionada
# if 'selected_voice' not in locals():
#     st.warning("Por favor, selecione uma voz para continuar.")
# else:
#     # Input de Texto
#     text_input = st.text_area("Insira o texto que deseja converter em áudio:", 
#                               value="""We were good, we were gold, Kind of dream that can't be sold, We were right 'til we weren't, Built a home and watched it burn""",
#                               height=150)

#     # Botão para Gerar Áudio
#     if st.button("Gerar Áudio"):
#         with st.spinner("Gerando áudio..."):
#             try:
#                 # Preparação do Texto
#                 if '|' in text_input:
#                     texts = text_input.split('|')
#                 else:
#                     texts = split_and_recombine_text(text_input)
                
#                 seed = int(time())
                
#                 # Diretório para salvar os resultados
#                 outpath = os.path.join("results", "longform", os.path.basename(selected_voice))
#                 os.makedirs(outpath, exist_ok=True)
                
#                 # Carregar amostras de voz
#                 voice_samples, conditioning_latents = load_voice(selected_voice)
                
#                 all_parts = []
#                 for j, text_part in enumerate(texts):
#                     gen = tts.tts_with_preset(
#                         text_part, 
#                         voice_samples=voice_samples, 
#                         conditioning_latents=conditioning_latents,
#                         preset="fast", 
#                         k=1, 
#                         use_deterministic_seed=seed
#                     )
#                     gen = gen.squeeze(0).cpu()
#                     part_path = os.path.join(outpath, f'{j}.wav')
#                     torchaudio.save(part_path, gen, 24000)
#                     all_parts.append(gen)
                
#                 # Concatenar todas as partes em um único áudio
#                 full_audio = torch.cat(all_parts, dim=-1)
#                 combined_path = os.path.join(outpath, 'combined.wav')
#                 torchaudio.save(combined_path, full_audio, 24000)
                
#                 # Exibir o áudio gerado
#                 st.audio(combined_path, format='audio/wav')
#                 st.success("Áudio gerado com sucesso!")
            
#             except Exception as e:
#                 st.error(f"Ocorreu um erro durante a geração do áudio: {e}")

if selected_voice:
    # Input de Texto
    st.header("Insira o Texto para Conversão em Áudio:")
    text_input = st.text_area("Texto:", 
                              value="""We were good, we were gold, Kind of dream that can't be sold, We were right 'til we weren't, Built a home and watched it burn""",
                              height=150)

    # Botão para Gerar Áudio
    if st.button("Gerar Áudio"):
        with st.spinner("Gerando áudio..."):
            try:
                # Preparação do Texto
                if '|' in text_input:
                    texts = text_input.split('|')
                else:
                    texts = split_and_recombine_text(text_input)
                
                seed = int(time())
                
                # Diretório para salvar os resultados
                outpath = os.path.join("results", "longform", os.path.basename(selected_voice))
                os.makedirs(outpath, exist_ok=True)
                
                # Carregar amostras de voz
                voice_samples, conditioning_latents = load_voice(selected_voice)
                
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
                
                # Concatenar todas as partes em um único áudio
                full_audio = torch.cat(all_parts, dim=-1)
                combined_path = os.path.join(outpath, 'combined.wav')
                torchaudio.save(combined_path, full_audio, 24000)

                with open(combined_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # Exibir o áudio gerado
                st.audio(combined_path, format='audio/wav')
                st.success("Áudio gerado com sucesso!")
            
            except Exception as e:
                st.error(f"Ocorreu um erro durante a geração do áudio: {e}")