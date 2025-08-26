from fastapi import WebSocket, WebSocketDisconnect, HTTPException,APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio
import base64
import json
from typing import Dict
import threading
import queue
import time


router = APIRouter(
    prefix="/Stream",
    tags=["Stream"]
)


# Cargar el modelo YOLO11
model = YOLO("./../AI/best.pt")

# Variables globales para el streaming
frame_queue = queue.Queue(maxsize=10)
processed_frame_queue = queue.Queue(maxsize=10)
is_processing = False
processing_thread = None

class StreamProcessor:
    def __init__(self):
        self.is_running = False
        self.current_frame = None
        
    def process_frames(self):
        """Procesa frames continuamente en un hilo separado"""
        while self.is_running:
            try:
                # Obtener frame del queue
                if not frame_queue.empty():
                    frame = frame_queue.get(timeout=1)
                    
                    # Procesar con YOLO
                    results = model(frame, verbose=False)
                    processed_frame = results[0].plot()
                    
                    # Almacenar frame procesado
                    if not processed_frame_queue.full():
                        processed_frame_queue.put(processed_frame)
                    else:
                        # Si está lleno, remover el frame más antiguo
                        try:
                            processed_frame_queue.get_nowait()
                            processed_frame_queue.put(processed_frame)
                        except:
                            pass
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error procesando frame: {e}")
                continue
                
        print("Processor thread stopped")

# Instancia del procesador
processor = StreamProcessor()

@router.post("/start_processing")
async def start_processing():
    """Inicia el procesamiento de frames"""
    global processing_thread, is_processing
    
    if not is_processing:
        processor.is_running = True
        processing_thread = threading.Thread(target=processor.process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        is_processing = True
        return {"message": "Processing started", "status": "running"}
    else:
        return {"message": "Processing already running", "status": "running"}

@router.post("/stop_processing")
async def stop_processing():
    """Detiene el procesamiento de frames"""
    global is_processing
    
    processor.is_running = False
    is_processing = False
    
    # Limpiar queues
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            break
    while not processed_frame_queue.empty():
        try:
            processed_frame_queue.get_nowait()
        except:
            break
            
    return {"message": "Processing stopped", "status": "stopped"}

@router.websocket("/stream_input")
async def websocket_input_endpoint(websocket: WebSocket):
    """WebSocket para recibir frames de entrada"""
    await websocket.accept()
    print("WebSocket input connected")
    
    try:
        while True:
            # Recibir datos del cliente
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decodificar el frame
            image_data = base64.b64decode(frame_data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Agregar frame al queue para procesamiento
                if not frame_queue.full():
                    frame_queue.put(frame)
                else:
                    # Si está lleno, remover el frame más antiguo
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(frame)
                    except:
                        pass
                        
    except WebSocketDisconnect:
        print("WebSocket input disconnected")
    except Exception as e:
        print(f"WebSocket input error: {e}")

@router.websocket("/stream_output")
async def websocket_output_endpoint(websocket: WebSocket):
    """WebSocket para enviar frames procesados"""
    await websocket.accept()
    print("WebSocket output connected")
    
    try:
        while True:
            # Obtener frame procesado
            if not processed_frame_queue.empty():
                processed_frame = processed_frame_queue.get()
                
                # Codificar frame como JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Enviar frame al cliente
                await websocket.send_text(json.dumps({
                    'image': frame_base64,
                    'timestamp': time.time()
                }))
            else:
                # Si no hay frames, esperar un poco
                await asyncio.sleep(0.03)  # ~30 FPS
                
    except WebSocketDisconnect:
        print("WebSocket output disconnected")
    except Exception as e:
        print(f"WebSocket output error: {e}")

def generate_stream():
    """Generador para HTTP streaming"""
    while True:
        if not processed_frame_queue.empty():
            processed_frame = processed_frame_queue.get()
            
            # Codificar frame como JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.03)  # ~30 FPS

@router.get("/stream_output_http")
async def stream_output_http():
    """Endpoint HTTP para streaming de frames procesados"""
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/process_frame")
async def process_single_frame(frame_data: dict):
    """Procesa un frame individual (para pruebas)"""
    try:
        # Decodificar el frame
        image_data = base64.b64decode(frame_data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Procesar con YOLO
        results = model(frame, verbose=False)
        processed_frame = results[0].plot()
        
        # Codificar resultado
        _, buffer = cv2.imencode('.jpg', processed_frame)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extraer información de detecciones
        detections = []
        for box in results[0].boxes:
            if box is not None:
                detection = {
                    'class_id': int(box.cls.item()),
                    'class_name': results[0].names[int(box.cls.item())],
                    'confidence': float(box.conf.item()),
                    'bbox': box.xyxy.tolist()[0]
                }
                detections.append(detection)
        
        return {
            'processed_image': result_base64,
            'detections': detections,
            'timestamp': time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/status")
async def get_status():
    """Obtiene el estado del procesamiento"""
    return {
        'processing': is_processing,
        'input_queue_size': frame_queue.qsize(),
        'output_queue_size': processed_frame_queue.qsize(),
        'model_loaded': True,
        'model_classes': list(model.names.values()) if hasattr(model, 'names') else []
    }

@router.get("/")
async def root():
    """Página principal con información de la API"""
    return {
        "message": "YOLO11 Live Stream Processing API",
        "endpoints": {
            "WebSocket Input": "/stream_input",
            "WebSocket Output": "/stream_output", 
            "HTTP Stream Output": "/stream_output_http",
            "Process Single Frame": "/process_frame",
            "Start Processing": "/start_processing",
            "Stop Processing": "/stop_processing",
            "Status": "/status"
        },
        "usage": {
            "1": "Start processing with POST /start_processing",
            "2": "Connect to WebSocket /stream_input to send frames",
            "3": "Connect to WebSocket /stream_output or GET /stream_output_http to receive processed frames",
            "4": "Send frames as base64 encoded JSON: {'image': 'base64_data'}"
        }
    }

@router.get("/demo")
async def demo_page():
    """Página de demostración"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO11 Live Stream Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            video, canvas { width: 400px; height: 300px; margin: 10px; border: 1px solid #ccc; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <h1>🎯 YOLO11 Live Stream Demo</h1>
        
        <div class="controls">
            <button onclick="startCamera()">📹 Iniciar Cámara</button>
            <button onclick="startProcessing()">🚀 Iniciar Procesamiento</button>
            <button onclick="stopProcessing()">⏹️ Detener Procesamiento</button>
            <button onclick="stopCamera()">📵 Detener Cámara</button>
        </div>
        
        <div id="status" class="status"></div>
        
        <div class="container">
            <div>
                <h3>📹 Cámara Original</h3>
                <video id="video" autoplay muted></video>
                <canvas id="canvas" style="display: none;"></canvas>
            </div>
            <div>
                <h3>🎯 Stream Procesado</h3>
                <canvas id="output"></canvas>
            </div>
        </div>
        
        <script>
            let video, canvas, ctx, outputCanvas, outputCtx;
            let inputSocket, outputSocket;
            let streaming = false;
            
            function showStatus(message, type = 'success') {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${type}`;
            }
            
            async function startCamera() {
                try {
                    video = document.getElementById('video');
                    canvas = document.getElementById('canvas');
                    ctx = canvas.getContext('2d');
                    outputCanvas = document.getElementById('output');
                    outputCtx = outputCanvas.getContext('2d');
                    
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    
                    video.addEventListener('loadedmetadata', () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        outputCanvas.width = video.videoWidth;
                        outputCanvas.height = video.videoHeight;
                        
                        showStatus('Cámara iniciada correctamente');
                        connectWebSockets();
                    });
                    
                } catch (err) {
                    showStatus('Error al acceder a la cámara: ' + err.message, 'error');
                }
            }
            
            function connectWebSockets() {
                // WebSocket para enviar frames
                inputSocket = new WebSocket('ws://localhost:8000/Stream/stream_input');
                inputSocket.onopen = () => {
                    console.log('Input WebSocket connected');
                    startSendingFrames();
                };
                
                // WebSocket para recibir frames procesados
                outputSocket = new WebSocket('ws://localhost:8000/Stream/stream_output');
                outputSocket.onopen = () => {
                    console.log('Output WebSocket connected');
                };
                
                outputSocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    displayProcessedFrame(data.image);
                };
            }
            
            function startSendingFrames() {
                streaming = true;
                sendFrame();
            }
            
            function sendFrame() {
                if (!streaming || !inputSocket || inputSocket.readyState !== WebSocket.OPEN) {
                    return;
                }
                
                // Capturar frame del video
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob((blob) => {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64 = reader.result.split(',')[1];
                        inputSocket.send(JSON.stringify({ image: base64 }));
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.8);
                
                // Programar siguiente frame (~30 FPS)
                setTimeout(sendFrame, 33);
            }
            
            function displayProcessedFrame(base64Image) {
                const img = new Image();
                img.onload = () => {
                    outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
                };
                img.src = 'data:image/jpeg;base64,' + base64Image;
            }
            
            async function startProcessing() {
                try {
                    const response = await fetch('http://localhost:8000/Stream/start_processing', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    showStatus('Procesamiento iniciado: ' + data.message);
                } catch (err) {
                    showStatus('Error al iniciar procesamiento: ' + err.message, 'error');
                }
            }
            
            async function stopProcessing() {
                try {
                    const response = await fetch('http://localhost:8000/Stream/stop_processing', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    showStatus('Procesamiento detenido: ' + data.message);
                } catch (err) {
                    showStatus('Error al detener procesamiento: ' + err.message, 'error');
                }
            }
            
            function stopCamera() {
                streaming = false;
                
                if (video && video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }
                
                if (inputSocket) inputSocket.close();
                if (outputSocket) outputSocket.close();
                
                showStatus('Cámara y conexiones detenidas');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
