from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, UnidentifiedImageError
import torch
import io
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Carrega o modelo TrOCR (fora do endpoint para evitar recarregar a cada requisição)
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    logger.info("Modelo TrOCR carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo TrOCR: {e}")
    raise RuntimeError("Falha ao carregar o modelo TrOCR.")

def recognize_text(image: Image.Image) -> str:
    """Reconhece o texto de uma imagem usando TrOCR."""
    try:
        # Converte a imagem para RGB (necessário para o TrOCR)
        image = image.convert("RGB")
        
        # Processa a imagem e gera os IDs do texto
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # Gera o texto usando o modelo
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # Decodifica os IDs para texto
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception as e:
        logger.error(f"Erro ao reconhecer texto: {e}")
        raise

@router.post("/ocr_placa2/")
async def ocr_plate(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna o texto da placa reconhecida."""
    try:
        # Verifica se o arquivo é uma imagem
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem.")
        
        # Lê a imagem
        image_data = await file.read()
        
        # Abre a imagem usando PIL
        try:
            image = Image.open(io.BytesIO(image_data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")
        
        # Reconhece o texto da imagem
        text = recognize_text(image)
        
        # Retorna o resultado
        return JSONResponse(content={"placa": text}, status_code=200)
    
    except HTTPException as http_err:
        # Captura exceções HTTP específicas e as repassa
        raise http_err
    except Exception as e:
        # Captura outros erros inesperados
        logger.error(f"Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar a imagem.")