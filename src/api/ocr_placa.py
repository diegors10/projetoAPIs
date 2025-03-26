from fastapi import APIRouter, File, UploadFile, HTTPException
import easyocr
import cv2
import numpy as np
import re
from typing import Optional

router = APIRouter()

# Inicializa o leitor EasyOCR para português e inglês
reader = easyocr.Reader(['pt', 'en'])

def preprocess_image(image_data: bytes, enhance_contrast: bool = True) -> np.ndarray:
    """ Aplica pré-processamento na imagem para otimizar a leitura da placa """
    
    # Converte os bytes para uma matriz OpenCV
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplica aumento de contraste
    if enhance_contrast:
        gray = cv2.equalizeHist(gray)

    # Aplica binarização adaptativa para destacar os caracteres
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Redução de ruído
    denoised = cv2.fastNlMeansDenoising(binarized, h=30)

    return denoised

def extract_license_plate(texts: list) -> Optional[str]:
    """ Filtra os textos reconhecidos para encontrar um formato de placa válido """

    # Formatos de placas no Brasil
    pattern_old = re.compile(r'^[A-Z]{3}-\d{4}$')      # Ex: ABC-1234
    pattern_new = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')  # Ex: ABC1D23 (placa Mercosul)

    for text in texts:
        clean_text = text.replace(" ", "").upper()  # Remove espaços desnecessários
        if pattern_old.match(clean_text) or pattern_new.match(clean_text):
            return clean_text  # Retorna a primeira placa válida encontrada

    return None  # Nenhuma placa encontrada

@router.post("/ocr_placa/")
async def ocr_endpoint(file: UploadFile = File(...), enhance_contrast: bool = True):
    """ Endpoint para processar imagens e extrair placas de carro """

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas arquivos de imagem são permitidos.")

    contents = await file.read()

    # Processa a imagem
    processed_image = preprocess_image(contents, enhance_contrast)

    # Salva temporariamente a imagem processada
    temp_filename = "temp_placa.png"
    cv2.imwrite(temp_filename, processed_image)

    # Realiza o OCR
    result = reader.readtext(temp_filename, detail=0)

    # Extrai a placa correta
    placa = extract_license_plate(result)

    if not placa:
        raise HTTPException(status_code=404, detail="Nenhuma placa reconhecida.")

    return {"placa": placa}
