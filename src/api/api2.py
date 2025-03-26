from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_api2():
    return {"message": "Esta é a API 2"}