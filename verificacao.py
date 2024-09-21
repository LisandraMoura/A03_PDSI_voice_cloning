import torchaudio

def verificar_audio(file_path):
    try:
        info = torchaudio.info(file_path)
        print(f"Arquivo: {file_path}")
        print(f"Taxa de Amostragem: {info.sample_rate} Hz")
        print(f"Canal: {info.num_channels}")
        print(f"Duração: {info.num_frames / info.sample_rate} segundos\n")
    except Exception as e:
        print(f"Erro ao verificar {file_path}: {e}\n")

verificar_audio("/home/lisamenezes/Searches/A03_PDSI_voice_cloning/tortoise-tts/tortoise/voices/miley/0.wav")
verificar_audio("/home/lisamenezes/Searches/A03_PDSI_voice_cloning/tortoise-tts/tortoise/voices/miley/1.wav")
