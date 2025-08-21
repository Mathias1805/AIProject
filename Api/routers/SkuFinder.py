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
from ultralytics import YOLO, solutions
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model_yolo = YOLO('./../AI/best.pt')

# Configuración de directorios
UPLOAD_DIR = "./Uploads/Images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    prefix="/SkuFinder",  # Prefijo para todos los endpoints en este router
    tags=["SkuFinder"]    # Etiqueta para la documentación
)

class FrameTrackingCropper:
    """Wrapper personalizado para trackear frames junto con crops"""
    
    def __init__(self, model, conf=0.8, crop_dir="./crops"):
        self.model = model
        self.conf = conf
        self.crop_dir = crop_dir
        self.frame_mapping = {}  # Mapeo de archivo crop -> número de frame
        self.crop_counter = 0

        os.makedirs(crop_dir, exist_ok=True)
        
    def process_frame(self, frame, frame_number):
        """Procesa un frame y guarda los crops con información del frame"""
        results = self.model(frame, conf=self.conf)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Obtener coordenadas de la caja
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Recortar la imagen
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:  # Verificar que el crop no esté vacío
                        # Crear nombre único para el crop
                        crop_filename = f"crop_{self.crop_counter:06d}.jpg"
                        crop_path = os.path.join(self.crop_dir, crop_filename)
                        
                        # Guardar el crop
                        cv2.imwrite(crop_path, crop)
                        
                        # Mapear el crop al frame
                        self.frame_mapping[crop_filename] = {
                            'frame_number': frame_number,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf[0]) if box.conf is not None else 0.0,
                            'class_id': int(box.cls[0]) if box.cls is not None else -1
                        }
                        
                        self.crop_counter += 1
        
        return results
    
    def save_frame_mapping(self, mapping_file):
        """Guardar el mapeo de frames en un archivo JSON"""
        with open(mapping_file, 'w') as f:
            json.dump(self.frame_mapping, f, indent=2)

@router.post("/compare/{video_id}")
async def process_video(
    video_id: int,
    image: UploadFile = File(...)
    ):
     
    try:
        # 1. Guardar la imagen subida
        file_path = os.path.join(UPLOAD_DIR, f"query_{video_id}.jpg")
        
        # Depuración: Verifica si el directorio existe y la ruta completa
        print(f"Intentando guardar la imagen en: {file_path}")
        print(f"Directorio existe: {os.path.exists(UPLOAD_DIR)}")
        
        # 2. Guardar la imagen subida
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Verificar si el archivo fue guardado correctamente
        if os.path.exists(file_path):
            print(f"Archivo guardado correctamente en: {file_path}")
        else:
            raise HTTPException(status_code=500, detail="Error al guardar la imagen")
        
        # 2. Preprocesar la imagen de consulta
        query_image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_features = model.encode_image(query_image)
        print("PASO 1 COMPLETE")
        # 3. Procesar recortes del video
        recortes_dir = f"./runs/detect/predict{video_id}"
        recortes_obj_dir = f"./runs/detect/predict{video_id}/objs"
        frame_mapping_file = f"./runs/detect/predict{video_id}/frame_mapping.json"
        
        print("PASO 1.1 COMPLETE")
        
        avi_files = []
        for file in os.listdir(recortes_dir):
            if file.endswith('.avi'):
                avi_files.append(os.path.join(recortes_dir, file))
        
        print(avi_files)
        
        print("PASO 2 COMPLETE")

        if not os.path.exists(recortes_obj_dir):
            VIDEO_PATH = str(avi_files[0])
            cap = cv2.VideoCapture(VIDEO_PATH)
            assert cap.isOpened(), f"No se pudo abrir el video: {VIDEO_PATH}"

            # Usar nuestro cropper personalizado con tracking de frames
            cropper = FrameTrackingCropper(
                model=model_yolo,
                conf=0.75,
                crop_dir=recortes_obj_dir
            )

            frame_number = 0
            print("Procesando video y extrayendo crops...")
            
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Procesar frame con tracking
                cropper.process_frame(frame, frame_number)
                frame_number += 1
                
                # Mostrar progreso cada 100 frames
                if frame_number % 100 == 0:
                    print(f"Procesados {frame_number} frames...")
            
            cap.release()
            
            # Guardar el mapeo de frames
            cropper.save_frame_mapping(frame_mapping_file)
            print(f"Mapeo de frames guardado en: {frame_mapping_file}")

        print("PASO 3 COMPLETE")

        # Cargar el mapeo de frames si existe
        frame_mapping = {}
        if os.path.exists(frame_mapping_file):
            with open(frame_mapping_file, 'r') as f:
                frame_mapping = json.load(f)

        # Comparación Manual
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
                try:
                    img = preprocess(Image.open(img_path))
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error procesando {file}: {e}")
                    continue
                    
            if batch_imgs:  # Solo procesar si hay imágenes válidas
                batch_imgs_tensor = torch.stack(batch_imgs).to(device)
                
                with torch.no_grad():
                    batch_features = model.encode_image(batch_imgs_tensor)
                    sims = (query_features @ batch_features.T).cpu().numpy().flatten()
                
                scores.extend(sims.tolist())
                names.extend(batch_files)

        # 4. Preparar resultados con información de frames
        results_with_frames = []
        for i, (name, score) in enumerate(zip(names, scores)):
            result = {
                'img': name,
                'similaridad': score
            }
            
            # Agregar información del frame si está disponible
            if name in frame_mapping:
                result.update({
                    'frame_number': frame_mapping[name]['frame_number'],
                    'bbox': frame_mapping[name]['bbox'],
                    'confidence': frame_mapping[name]['confidence'],
                    'class_id': frame_mapping[name]['class_id']
                })
            
            results_with_frames.append(result)
        
        print("PASO 4 COMPLETE")

        # Ordenar por similaridad
        results_with_frames.sort(key=lambda x: x['similaridad'], reverse=True)
        top_results = results_with_frames[:10]

        return JSONResponse({
            "video_id": video_id,
            "query_image": file_path,
            "total_crops": len(files),
            "total_frames_processed": max([frame_mapping[name]['frame_number'] for name in frame_mapping.keys()]) + 1 if frame_mapping else 0,
            "top_matches": top_results,
            "frame_mapping_file": frame_mapping_file,
            "status": "processed"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))