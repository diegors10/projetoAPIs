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
import base64  # Para codificar áudio em base64
import time
from dotenv import load_dotenv
load_dotenv()

# Caminhos do FFmpeg (necessário para manipular áudio com pydub)
FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\ffmpeg\\bin\\ffprobe.exe"

# Token para acessar o modelo do Hugging Face
TOKEN_HF = os.getenv("API_KEY_HUG")
if not TOKEN_HF:
    raise ValueError("Token do Hugging Face não fornecido.")

# Configura o caminho do ffmpeg para o pydub e o PATH do sistema
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instancia o roteador da API
router = APIRouter()

# Diretórios para uploads e saídas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Armazena o status das tarefas em memória
processing_tasks = {}

# Função para extrair o áudio de um vídeo
def extract_audio(video_path: str, audio_path: str):
    try:
        logger.info(f"Extraindo áudio de {video_path}")
        ffmpeg.input(video_path).output(audio_path, format='mp3', acodec='libmp3lame').run(overwrite_output=True)
        logger.info(f"Áudio salvo em {audio_path}")
    except Exception as e:
        logger.error(f"Erro ao extrair áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao extrair áudio: {str(e)}")

# Função de diarização do áudio
def diarize_audio(audio_path: str, task_id: str):
    try:
        logger.info(f"Iniciando diarização de {audio_path}")
        start_time = time.time()

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=TOKEN_HF)
        if not pipeline:
            raise ValueError("Falha ao carregar o pipeline.")

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

        elapsed = time.time() - start_time
        logger.info(f"Arquivos gerados: {output_files} em {elapsed:.2f} segundos")

        processing_tasks[task_id] = {
            "status": "completed",
            "files": output_files,
            "duration": elapsed
        }
    except Exception as e:
        logger.error(f"Erro na diarização: {e}")
        processing_tasks[task_id] = {"status": "failed", "error": str(e)}

@router.post("/extrator_audio/")
async def extract_audio_from_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ["mp4", "mkv", "avi", "mov"]:
            raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")

        task_id = str(uuid.uuid4())
        video_filename = f"{task_id}.{file_ext}"
        audio_filename = f"{task_id}.mp3"

        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_audio(video_path, audio_path)

        processing_tasks[task_id] = {"status": "processing"}
        background_tasks.add_task(diarize_audio, audio_path, task_id)

        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        logger.error(f"Erro no endpoint /extrator_audio/: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/extrator_audio/status/{task_id}")
async def get_status(task_id: str):
    task_status = processing_tasks.get(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada")

    if task_status["status"] == "completed":
        files_with_base64 = []

        for file_path in task_status["files"]:
            with open(file_path, "rb") as audio_file:
                audio_content = audio_file.read()
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")

            relative_path = file_path.replace(BASE_DIR + os.sep, "").replace("\\", "/")
            
            files_with_base64.append({
                "file_url": f"http://localhost:5000/{relative_path}",
                "file_base64": audio_base64
            })

        return {
            "task_id": task_id,
            "status": "completed",
            "duration_seconds": task_status.get("duration", None),
            "files": files_with_base64
        }

    elif task_status["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": task_status["error"]
        }

    else:
        return {"task_id": task_id, "status": "processing"}
