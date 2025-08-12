import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import cv2
from ultralytics import YOLO,solutions

device = "cuda" 
model, preprocess = clip.load("ViT-B/32", device=device)
model_yolo = YOLO('./../AI/best.pt')

# Configuración de directorios
UPLOAD_DIR = "/Api/Uploads/Images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    prefix="/SkuFinder",  # Prefijo para todos los endpoints en este router
    tags=["SkuFinder"]    # Etiqueta para la documentación
)

@router.post("/compare/{video_id}")
async def process_video(
    video_id: int,
    image: UploadFile = File(...)
):
    try:
        # 1. Guardar la imagen subida
        file_path = os.path.join(UPLOAD_DIR, f"query_{video_id}.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # 2. Preprocesar la imagen de consulta
        query_image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_features = model.encode_image(query_image)
        
        # 3. Procesar recortes del video
        recortes_dir = f"./runs/detect/predict{video_id}"
        recortes_obj_dir = f"./runs/detect/predict{video_id}/objs"

        avi_files = []
        for file in os.listdir(recortes_dir):
            if file.endswith('.avi'):
                avi_files.append(os.path.join(recortes_dir, file))

        if len(avi_files) == 1:
            mp4_file_avi = str(avi_files[0])

            mp4_file_mp4=mp4_file_avi.replace(".avi",".mp4")

            os.rename(str(avi_files[0]), mp4_file_mp4)

        mp4_files = []
        for file in os.listdir(recortes_dir):
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(recortes_dir, file))    

        if not os.path.exists(recortes_obj_dir):
            VIDEO_PATH = str(mp4_files[0])
            cap = cv2.VideoCapture(VIDEO_PATH)
            assert cap.isOpened(), f"No se pudo abrir el video: {VIDEO_PATH}"

            cropper = solutions.ObjectCropper(
                show=False,               
                model=model_yolo,
                conf=0.8,
                crop_dir=recortes_obj_dir
            )

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cropper(frame)
            cap.release()

        batch_size = 1000
        files = [f for f in os.listdir(recortes_obj_dir) if f.endswith(".jpg")]
        scores = []
        names = []

        # Procesar en batches
        for i in tqdm(range(0, len(files), batch_size), desc="Comparando imágenes"):
            batch_files = files[i:i+batch_size]
            batch_imgs = []
            
            for file in batch_files:
                img_path = os.path.join(recortes_obj_dir, file)
                img = preprocess(Image.open(img_path))
                batch_imgs.append(img)
                
            batch_imgs_tensor = torch.stack(batch_imgs).to(device)
            
            with torch.no_grad():
                batch_features = model.encode_image(batch_imgs_tensor)
                sims = (query_features @ batch_features.T).cpu().numpy().flatten()
            
            scores.extend(sims.tolist())
            names.extend(batch_files)

        # 4. Preparar resultados
        df = pd.DataFrame({'img': names, 'similaridad': scores})
        df = df.sort_values('similaridad', ascending=False)
        top_results = df.head(10).to_dict('records')

        return JSONResponse({
            "video_id": video_id,
            "query_image": file_path,
            "top_matches": top_results,
            "status": "processed"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

