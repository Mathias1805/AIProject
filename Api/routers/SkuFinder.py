import shutil
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import cv2
from ultralytics import YOLO
import json
import time
from typing import Dict, Optional
from enum import Enum

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model_yolo = YOLO('./../AI/best.pt')

# Configuración de directorios
UPLOAD_DIR = "./Uploads/Images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Estados de procesamiento
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Store global para tracking de tareas
processing_tasks: Dict[str, Dict] = {}

router = APIRouter(
    prefix="/SkuFinder",
    tags=["SkuFinder"]
)

class FrameTrackingCropper:
    """Wrapper personalizado para trackear frames junto con crops"""
    
    def __init__(self, model, conf=0.8, crop_dir="./crops"):
        self.model = model
        self.conf = conf
        self.crop_dir = crop_dir
        self.frame_mapping = {}
        self.crop_counter = 0
        os.makedirs(crop_dir, exist_ok=True)
        
    def process_frame(self, frame, frame_number):
        """Procesa un frame y guarda los crops con información del frame"""
        results = self.model(frame, conf=self.conf)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        crop_filename = f"crop_{self.crop_counter:06d}.jpg"
                        crop_path = os.path.join(self.crop_dir, crop_filename)
                        cv2.imwrite(crop_path, crop)
                        
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

def update_task_progress(task_id: str, progress: float, status: str, message: str = ""):
    """Actualizar progreso de una tarea"""
    if task_id in processing_tasks:
        processing_tasks[task_id].update({
            'progress': progress,
            'status': status,
            'message': message,
            'updated_at': time.time()
        })

async def process_video_async(
    task_id: str,
    video_id: int,
    file_path: str,
    query_features: torch.Tensor
):
    """Función asíncrona principal de procesamiento"""
    try:
        update_task_progress(task_id, 5, ProcessingStatus.PROCESSING, "Iniciando procesamiento")
        
        # Configurar directorios
        recortes_dir = f"./runs/detect/predict{video_id}"
        recortes_obj_dir = f"./runs/detect/predict{video_id}/objs"
        frame_mapping_file = f"./runs/detect/predict{video_id}/frame_mapping.json"
        
        update_task_progress(task_id, 10, ProcessingStatus.PROCESSING, "Configurando directorios")
        
        # Buscar archivos de video
        avi_files = [os.path.join(recortes_dir, f) for f in os.listdir(recortes_dir) if f.endswith('.avi')]
        
        if not avi_files:
            raise Exception("No se encontraron archivos de video")
        
        update_task_progress(task_id, 15, ProcessingStatus.PROCESSING, "Archivo de video encontrado")
        
        # Procesar video si no existen los crops
        if not os.path.exists(recortes_obj_dir):
            await process_video_crops(task_id, avi_files[0], recortes_obj_dir, frame_mapping_file)
        else:
            update_task_progress(task_id, 40, ProcessingStatus.PROCESSING, "Crops existentes encontrados")
        
        # Cargar mapeo de frames
        frame_mapping = {}
        if os.path.exists(frame_mapping_file):
            with open(frame_mapping_file, 'r') as f:
                frame_mapping = json.load(f)
        
        update_task_progress(task_id, 45, ProcessingStatus.PROCESSING, "Iniciando comparación de imágenes")
        
        # Comparación de imágenes (en hilo separado para no bloquear)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results_with_frames = await loop.run_in_executor(
                executor, 
                compare_images_batch, 
                task_id, 
                recortes_obj_dir, 
                query_features, 
                frame_mapping
            )
        
        update_task_progress(task_id, 80, ProcessingStatus.PROCESSING, "Extrayendo frames de resultados")
        
        # Extraer frames de los top matches
        timestamp = int(time.time())
        results_dir = await extract_frames_async(task_id, video_id, timestamp, recortes_dir, results_with_frames, avi_files[0])
        
        # Resultado final
        result = {
            "task_id": task_id,
            "video_id": video_id,
            "query_image": file_path,
            "total_crops": len([f for f in os.listdir(recortes_obj_dir) if f.endswith(".jpg")]) if os.path.exists(recortes_obj_dir) else 0,
            "top_matches": results_with_frames[:10],
            "frame_mapping_file": frame_mapping_file,
            "extracted_frames": {
                "results_directory": results_dir,
                "metadata_file": f"{results_dir}/extraction_metadata.json"
            },
            "status": "completed",
            "completed_at": time.time()
        }
        
        processing_tasks[task_id]['result'] = result
        update_task_progress(task_id, 100, ProcessingStatus.COMPLETED, "Procesamiento completado exitosamente")
        
    except Exception as e:
        error_msg = f"Error durante el procesamiento: {str(e)}"
        update_task_progress(task_id, 0, ProcessingStatus.FAILED, error_msg)
        processing_tasks[task_id]['error'] = error_msg

async def process_video_crops(task_id: str, video_path: str, crops_dir: str, mapping_file: str):
    """Procesar video y extraer crops de manera asíncrona"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cropper = FrameTrackingCropper(model=model_yolo, conf=0.75, crop_dir=crops_dir)
    
    frame_number = 0
    
    # Usar ThreadPoolExecutor para procesamiento intensivo
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame en hilo separado
            await loop.run_in_executor(executor, cropper.process_frame, frame, frame_number)
            
            frame_number += 1
            
            # Actualizar progreso cada 50 frames
            if frame_number % 50 == 0:
                progress = 15 + (frame_number / total_frames) * 25  # 15% a 40%
                update_task_progress(task_id, progress, ProcessingStatus.PROCESSING, f"Procesados {frame_number}/{total_frames} frames")
                await asyncio.sleep(0.01)  # Pequeña pausa para no saturar
    
    cap.release()
    cropper.save_frame_mapping(mapping_file)
    update_task_progress(task_id, 40, ProcessingStatus.PROCESSING, "Crops extraídos exitosamente")

def compare_images_batch(task_id: str, crops_dir: str, query_features: torch.Tensor, frame_mapping: dict):
    """Comparar imágenes por lotes (ejecutado en ThreadPoolExecutor)"""
    batch_size = 1000
    files = [f for f in os.listdir(crops_dir) if f.endswith(".jpg")]
    scores = []
    names = []
    
    total_batches = len(files) // batch_size + (1 if len(files) % batch_size > 0 else 0)
    
    for batch_idx, i in enumerate(range(0, len(files), batch_size)):
        batch_files = files[i:i+batch_size]
        batch_imgs = []
        
        for file in batch_files:
            img_path = os.path.join(crops_dir, file)
            try:
                img = preprocess(Image.open(img_path))
                batch_imgs.append(img)
            except Exception as e:
                continue
                
        if batch_imgs:
            batch_imgs_tensor = torch.stack(batch_imgs).to(device)
            
            with torch.no_grad():
                batch_features = model.encode_image(batch_imgs_tensor)
                sims = (query_features @ batch_features.T).cpu().numpy().flatten()
            
            scores.extend(sims.tolist())
            names.extend(batch_files)
        
        # Actualizar progreso
        progress = 45 + (batch_idx / total_batches) * 30  # 45% a 75%
        update_task_progress(task_id, progress, ProcessingStatus.PROCESSING, f"Comparando lote {batch_idx + 1}/{total_batches}")
    
    # Preparar resultados con información de frames
    results_with_frames = []
    for name, score in zip(names, scores):
        result = {'img': name, 'similaridad': score}
        if name in frame_mapping:
            result.update({
                'frame_number': frame_mapping[name]['frame_number'],
                'bbox': frame_mapping[name]['bbox'],
                'confidence': frame_mapping[name]['confidence'],
                'class_id': frame_mapping[name]['class_id']
            })
        results_with_frames.append(result)
    
    results_with_frames.sort(key=lambda x: x['similaridad'], reverse=True)
    return results_with_frames

async def extract_frames_async(task_id: str, video_id: int, timestamp: int, recortes_dir: str, results: list, video_path: str):
    """Extraer frames de manera asíncrona"""
    results_dir = f"{recortes_dir}/results/video_{video_id}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Extraer frames únicos
    unique_frames = set()
    for match in results[:10]:
        if 'frame_number' in match:
            unique_frames.add(match['frame_number'])
    
    unique_frames = sorted(list(unique_frames))
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        extracted_frames = await loop.run_in_executor(
            executor, 
            extract_frames_sync, 
            video_path, 
            unique_frames, 
            results_dir
        )
    
    # Crear metadatos
    metadata = {
        "extraction_info": {
            "total_matches": len(results[:10]),
            "unique_frames_extracted": len(extracted_frames),
            "results_directory": results_dir
        },
        "extracted_frames": extracted_frames,
        "matches_detail": results[:10]
    }
    
    metadata_path = os.path.join(results_dir, "extraction_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results_dir

def extract_frames_sync(video_path: str, unique_frames: list, results_dir: str):
    """Extraer frames de manera síncrona (para ThreadPoolExecutor)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    extracted_frames = []
    current_frame = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < len(unique_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame in unique_frames:
            frame_filename = f"frame_{current_frame:06d}.jpg"
            frame_path = os.path.join(results_dir, frame_filename)
            
            if cv2.imwrite(frame_path, frame):
                extracted_frames.append({
                    'frame_number': current_frame,
                    'filename': frame_filename,
                    'path': frame_path
                })
                extracted_count += 1
        
        current_frame += 1
    
    cap.release()
    return extracted_frames

# ENDPOINTS

@router.post("/compare/{video_id}")
async def process_video_non_blocking(
    video_id: int,
    image: UploadFile = File(...)
):
    """Endpoint no bloqueante que inicia el procesamiento y retorna task_id"""
    try:
        # Generar ID único para la tarea
        task_id = f"task_{video_id}_{int(time.time())}_{hash(image.filename) % 10000}"
        
        # Guardar imagen
        timestamp = int(time.time())
        file_path = os.path.join(UPLOAD_DIR, f"query_{video_id}_{timestamp}.jpg")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Error al guardar la imagen")
        
        # Preprocesar imagen de consulta
        query_image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_features = model.encode_image(query_image)
        
        # Registrar tarea
        processing_tasks[task_id] = {
            'task_id': task_id,
            'video_id': video_id,
            'status': ProcessingStatus.PENDING,
            'progress': 0,
            'message': 'Tarea iniciada',
            'started_at': time.time(),
            'updated_at': time.time(),
            'query_image_path': file_path
        }
        
        # Iniciar procesamiento asíncrono
        asyncio.create_task(process_video_async(task_id, video_id, file_path, query_features))
        
        return JSONResponse({
            "task_id": task_id,
            "video_id": video_id,
            "status": "accepted",
            "message": "Procesamiento iniciado. Use el task_id para verificar el progreso.",
            "check_status_url": f"/SkuFinder/status/{task_id}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Obtener el estado de una tarea de procesamiento"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task ID no encontrado")
    
    task_info = processing_tasks[task_id].copy()
    
    # Si está completada, incluir resultados
    if task_info['status'] == ProcessingStatus.COMPLETED and 'result' in processing_tasks[task_id]:
        task_info['result'] = processing_tasks[task_id]['result']
    
    return JSONResponse(task_info)

@router.get("/tasks")
async def list_all_tasks():
    """Listar todas las tareas de procesamiento"""
    tasks = []
    for task_id, task_info in processing_tasks.items():
        task_summary = {
            'task_id': task_id,
            'video_id': task_info.get('video_id'),
            'status': task_info.get('status'),
            'progress': task_info.get('progress'),
            'started_at': task_info.get('started_at'),
            'updated_at': task_info.get('updated_at')
        }
        tasks.append(task_summary)
    
    return JSONResponse({
        "total_tasks": len(tasks),
        "tasks": sorted(tasks, key=lambda x: x['updated_at'], reverse=True)
    })

@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Eliminar una tarea del registro"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task ID no encontrado")
    
    deleted_task = processing_tasks.pop(task_id)
    return JSONResponse({
        "message": f"Tarea {task_id} eliminada",
        "deleted_task": {
            "task_id": task_id,
            "status": deleted_task.get('status'),
            "video_id": deleted_task.get('video_id')
        }
    })
