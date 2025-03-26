from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import logging
from pydub import AudioSegment
import ffmpeg
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch
import base64  # Importar a biblioteca base64 para codificação

# Configurações
FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\ffmpeg\\bin\\ffprobe.exe"
TOKEN_HF = "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"

if not TOKEN_HF:
    raise ValueError("Token do Hugging Face não fornecido. Defina a variável TOKEN_HF.")

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Dicionário para armazenar o status das tarefas
processing_tasks = {}

def extract_audio(video_path: str, audio_path: str):
    try:
        logger.info(f"Extraindo áudio do vídeo: {video_path}")
        ffmpeg.input(video_path).output(audio_path, format='mp3', acodec='libmp3lame').run(overwrite_output=True)
        logger.info(f"Áudio extraído e salvo em: {audio_path}")
    except Exception as e:
        logger.error(f"Erro ao extrair áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao extrair áudio: {str(e)}")

def diarize_audio(audio_path: str, task_id: str):
    try:
        logger.info(f"Iniciando diarização do áudio: {audio_path}")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=TOKEN_HF
        )
        
        if not pipeline:
            raise ValueError("Falha ao carregar o pipeline de diarização.")
        
        diarization = pipeline(audio_path)
        
        audio = AudioSegment.from_mp3(audio_path)
        speaker_segments = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        output_files = []
        for speaker, segments in speaker_segments.items():
            speaker_audio = sum(segments)
            output_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_{speaker}.mp3")
            speaker_audio.export(output_path, format="mp3")
            output_files.append(output_path)
        
        logger.info(f"Arquivos gerados: {output_files}")
        processing_tasks[task_id] = {"status": "completed", "files": output_files}
    except Exception as e:
        logger.error(f"Erro ao diarizar áudio: {e}")
        processing_tasks[task_id] = {"status": "failed", "error": str(e)}

@router.post("/extrator_audio/")
async def extract_audio_from_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        file_ext = file.filename.split(".")[-1]
        if file_ext not in ["mp4", "mkv", "avi", "mov"]:
            raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")
        
        # Gerar um ID único para a tarefa
        task_id = str(uuid.uuid4())
        
        # Gerar nomes únicos para os arquivos
        video_filename = f"{task_id}.{file_ext}"
        audio_filename = f"{task_id}.mp3"
        
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)
        
        # Salvar o arquivo de vídeo
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extrair áudio do vídeo
        extract_audio(video_path, audio_path)
        
        # Iniciar a diarização em segundo plano
        background_tasks.add_task(diarize_audio, audio_path, task_id)
        
        # Retornar o ID da tarefa
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Erro no endpoint /extrator_audio/: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/extrator_audio/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada")
    
    task_status = processing_tasks[task_id]
    if task_status["status"] == "completed":
        # Lista para armazenar os arquivos com links e Base64
        files_with_base64 = []
        
        for file_path in task_status["files"]:
            # Ler o conteúdo do arquivo
            with open(file_path, "rb") as audio_file:
                audio_content = audio_file.read()
            
            # Codificar o conteúdo em Base64
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")
            
            # Adicionar à lista
            files_with_base64.append({
                "file_url": f"http://localhost:5000/{file_path}",  # Link para download
                "file_base64": audio_base64  # Conteúdo em Base64
            })
        
        return {
            "task_id": task_id,
            "status": "completed",
            "files": files_with_base64
        }
    elif task_status["status"] == "failed":
        return {"task_id": task_id, "status": "failed", "error": task_status["error"]}
    else:
        return {"task_id": task_id, "status": "processing"}