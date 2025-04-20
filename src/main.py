from fastapi import FastAPI
from .api.extrator_audio import router as extrator_audio_router
from .api.ocr_placa import router as ocr_placa_router
from .api.ocr_placa2 import router as ocr_placa_router2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens, ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta as rotas das APIs
app.include_router(extrator_audio_router, prefix="/api")  # Prefixo alterado para /api
app.include_router(ocr_placa_router, prefix="/api")  # Prefixo alterado para /api
app.include_router(ocr_placa_router2, prefix="/api")  # Prefixo alterado para /api


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)