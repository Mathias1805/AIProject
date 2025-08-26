from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from routers.SkuFinder import router as SkuFinder_router
from routers.Stream import router as Stream_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Incluir el router
app.include_router(SkuFinder_router)
app.include_router(Stream_router)

def process_video_simple(video_path, conf=0.5):
    # Cargar tu modelo entrenado
    model = YOLO('./AI/best.pt')
    # Procesa un video completo de forma automática
    print(f"🎥 Procesando video: {video_path}")
    # Procesar video completo
    results = model(video_path, conf=conf, save=True)
    print("✅ Video procesado")
    return results

@app.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    # Validar que sea un archivo de video
    if not video.content_type.startswith("video/"):
        return JSONResponse(
            status_code=400, 
            content={"error": "El archivo debe ser un video"}
        )
    
    # Obtener la extensión del archivo original
    original_extension = Path(video.filename).suffix
    
    # Crear el nuevo nombre con el formato: video + año + fecha + hora + segundo + longitud
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")  # año + mes + día + hora + minuto + segundo
    
    # Obtener el tamaño del archivo
    file_size = video.size
    
    # Crear el nuevo nombre
    new_filename = f"video{timestamp}{file_size}{original_extension}"
    
    # Guardar el archivo
    upload_dir = Path("Uploads/Videos")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / new_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

        
    # Procesar el video
    process_video_simple(str(file_path))

    return {
        "original_filename": video.filename,
        "new_filename": new_filename,
        "content_type": video.content_type,
        "size": video.size,
        "timestamp": timestamp,
        "file_path": str(file_path),
        "message": "Video subido y procesado exitosamente"
    }
