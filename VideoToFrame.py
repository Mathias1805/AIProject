#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para extraer frames de videos con diferentes estrategias
Optimizado para análisis de productos en tiendas
"""

import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json


class VideoFrameExtractor:
    def __init__(self, video_path, output_dir="frames", quality_threshold=30):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.quality_threshold = quality_threshold
        self.stats = {
            'total_frames': 0,
            'extracted_frames': 0,
            'blurry_frames': 0,
            'duration': 0,
            'fps': 0
        }
        
    def create_output_directory(self):
        """Crear directorio de salida si no existe"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio de salida: {self.output_dir}")
        
    def calculate_blur_score(self, frame):
        """Calcular score de nitidez usando Laplacian"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score
    
    def detect_scene_change(self, prev_frame, curr_frame, threshold=0.3):
        """Detectar cambios significativos entre frames"""
        if prev_frame is None:
            return True
            
        # Convertir a escala de grises y redimensionar para eficiencia
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        prev_small = cv2.resize(prev_gray, (64, 64))
        curr_small = cv2.resize(curr_gray, (64, 64))
        
        # Calcular diferencia
        diff = cv2.absdiff(prev_small, curr_small)
        change_ratio = np.sum(diff > 30) / (64 * 64)
        
        return change_ratio > threshold

    def extract_all_frames(self, interval_seconds=0.5):
        """Extraer frames a intervalos regulares"""
        print("🎬 Extrayendo frames a intervalos regulares...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        # Obtener información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.stats.update({
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        })
        
        print(f"📊 Video: {duration:.2f}s, {fps:.2f} FPS, {total_frames} frames")
        
        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Calcular calidad del frame
                blur_score = self.calculate_blur_score(frame)
                
                if blur_score >= self.quality_threshold:
                    timestamp = frame_count / fps
                    filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}_q{blur_score:.1f}.jpg"
                    filepath = self.output_dir / filename
                    
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_count += 1
                    
                    if extracted_count % 50 == 0:
                        print(f"✅ Extraídos {extracted_count} frames...")
                else:
                    self.stats['blurry_frames'] += 1
            
            frame_count += 1
        
        cap.release()
        self.stats['extracted_frames'] = extracted_count
        print(f"🎯 Completado: {extracted_count} frames extraídos")

    def extract_scene_changes(self, min_interval_seconds=1.0):
        """Extraer frames solo cuando hay cambios de escena"""
        print("🎬 Extrayendo frames basado en cambios de escena...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.stats.update({
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        })
        
        min_frame_interval = int(fps * min_interval_seconds)
        
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        last_extracted_frame = -min_frame_interval
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Verificar intervalo mínimo
            if frame_count - last_extracted_frame < min_frame_interval:
                frame_count += 1
                continue
            
            # Detectar cambio de escena
            if self.detect_scene_change(prev_frame, frame):
                blur_score = self.calculate_blur_score(frame)
                
                if blur_score >= self.quality_threshold:
                    timestamp = frame_count / fps
                    filename = f"scene_{extracted_count:06d}_t{timestamp:.2f}_q{blur_score:.1f}.jpg"
                    filepath = self.output_dir / filename
                    
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_count += 1
                    last_extracted_frame = frame_count
                    
                    print(f"📸 Cambio detectado en {timestamp:.2f}s (calidad: {blur_score:.1f})")
                else:
                    self.stats['blurry_frames'] += 1
            
            prev_frame = frame.copy()
            frame_count += 1
        
        cap.release()
        self.stats['extracted_frames'] = extracted_count
        print(f"🎯 Completado: {extracted_count} frames de cambios de escena")

    def extract_adaptive(self, base_interval=1.0, max_interval=5.0):
        """Extracción adaptiva basada en movimiento y calidad"""
        print("🎬 Extrayendo frames con método adaptivo...")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.stats.update({
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        })
        
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        last_extracted_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            blur_score = self.calculate_blur_score(frame)
            
            # Calcular intervalo dinámico basado en movimiento
            if prev_frame is not None:
                scene_change = self.detect_scene_change(prev_frame, frame, threshold=0.1)
                
                if scene_change:
                    dynamic_interval = base_interval
                else:
                    dynamic_interval = min(max_interval, base_interval * 2)
            else:
                dynamic_interval = base_interval
            
            # Decidir si extraer el frame
            time_since_last = timestamp - (last_extracted_frame / fps)
            should_extract = (
                time_since_last >= dynamic_interval and 
                blur_score >= self.quality_threshold
            )
            
            if should_extract:
                filename = f"adaptive_{extracted_count:06d}_t{timestamp:.2f}_q{blur_score:.1f}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
                last_extracted_frame = frame_count
                
                if extracted_count % 25 == 0:
                    print(f"✅ Extraídos {extracted_count} frames adaptativos...")
            elif blur_score < self.quality_threshold:
                self.stats['blurry_frames'] += 1
            
            prev_frame = frame.copy()
            frame_count += 1
        
        cap.release()
        self.stats['extracted_frames'] = extracted_count
        print(f"🎯 Completado: {extracted_count} frames adaptativos")

    def save_metadata(self):
        """Guardar metadatos de la extracción"""
        metadata = {
            'video_path': str(self.video_path),
            'extraction_date': datetime.now().isoformat(),
            'stats': self.stats,
            'quality_threshold': self.quality_threshold
        }
        
        metadata_path = self.output_dir / 'extraction_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadatos guardados en: {metadata_path}")

    def print_summary(self):
        """Mostrar resumen de la extracción"""
        print("\n" + "="*50)
        print("📊 RESUMEN DE EXTRACCIÓN")
        print("="*50)
        print(f"Video original: {self.stats['duration']:.2f} segundos")
        print(f"FPS del video: {self.stats['fps']:.2f}")
        print(f"Frames totales: {self.stats['total_frames']:,}")
        print(f"Frames extraídos: {self.stats['extracted_frames']:,}")
        print(f"Frames borrosos descartados: {self.stats['blurry_frames']:,}")
        print(f"Tasa de extracción: {(self.stats['extracted_frames']/self.stats['total_frames']*100):.2f}%")
        print(f"Umbral de calidad: {self.quality_threshold}")
        print("="*50)


def mostrar_ayuda():
    """Muestra ayuda detallada del script"""
    help_text = """
🎬 EXTRACTOR DE FRAMES PARA ANÁLISIS DE PRODUCTOS
================================================

DESCRIPCIÓN:
    Script para extraer frames de videos con filtrado inteligente de calidad.
    Optimizado para análisis de productos en videos de tiendas.

USO BÁSICO:
    python extract_frames.py <video_path> [opciones]

EJEMPLOS PRÁCTICOS:
    # Extracción adaptiva (recomendado)
    python extract_frames.py tienda.mp4 -o frames_productos -m adaptive -q 40

    # Cada 2 segundos con alta calidad
    python extract_frames.py tienda.mp4 -m interval -i 2.0 -q 50

    # Solo cambios de escena importantes
    python extract_frames.py tienda.mp4 -m scene -i 1.5 -q 35

    # Procesamiento rápido con calidad básica
    python extract_frames.py tienda.mp4 -m interval -i 3.0 -q 25

PARÁMETROS:
    video_path              Ruta del archivo de video (requerido)
    
    -o, --output           Directorio de salida (default: 'frames')
    -m, --method           Método de extracción:
                            • interval  : Intervalos fijos
                            • scene     : Cambios de escena
                            • adaptive  : Adaptivo (recomendado)
    -q, --quality          Umbral de calidad 0-100 (default: 30)
                            • 20-30: Calidad básica, más frames
                            • 30-50: Calidad media (recomendado)
                            • 50+  : Alta calidad, menos frames
    -i, --interval         Intervalo base en segundos (default: 1.0)
    -h, --help            Mostrar esta ayuda

MÉTODOS EXPLICADOS:
    🔄 INTERVAL (intervalos):
        - Extrae frames cada X segundos
        - Predecible, good para videos uniformes
        - Mejor para: videos con ritmo constante

    🎭 SCENE (cambios de escena):
        - Solo extrae cuando detecta cambios significativos
        - Ideal para videos con transiciones claras
        - Mejor para: cuando el vendedor cambia de producto

    🧠 ADAPTIVE (adaptivo):
        - Combina intervalos y detección de cambios
        - Se adapta automáticamente al contenido
        - Mejor para: mayoría de casos, especialmente productos

SALIDA:
    - Frames numerados con timestamp y calidad
    - Archivo metadata.json con estadísticas
    - Nombres de archivo informativos

RECOMENDACIONES PARA PRODUCTOS:
    • Videos de tienda: -m adaptive -q 40 -i 1.5
    • Videos borrosos: -m adaptive -q 25 -i 2.0
    • Videos HD nítidos: -m scene -q 50 -i 1.0
    • Procesamiento rápido: -m interval -q 30 -i 3.0

REQUISITOS:
    pip install opencv-python numpy

FORMATOS SOPORTADOS:
    MP4, AVI, MOV, MKV, WMV, FLV
================================================
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description='Extraer frames de video para análisis de productos',
        add_help=False  # Deshabilitamos el -h automático
    )
    
    # Parámetro posicional opcional para permitir solo --help
    parser.add_argument('video_path', nargs='?', help='Ruta del archivo de video')
    parser.add_argument('-o', '--output', default='frames', help='Directorio de salida')
    parser.add_argument('-m', '--method', choices=['interval', 'scene', 'adaptive'], 
                       default='adaptive', help='Método de extracción')
    parser.add_argument('-q', '--quality', type=int, default=30, 
                       help='Umbral mínimo de calidad (nitidez)')
    parser.add_argument('-i', '--interval', type=float, default=1.0, 
                       help='Intervalo base en segundos')
    parser.add_argument('-h', '--help', action='store_true', 
                       help='Mostrar ayuda detallada')
    
    args = parser.parse_args()
    
    # Mostrar ayuda si se solicita o no hay video_path
    if args.help or not args.video_path:
        mostrar_ayuda()
        return
    
    # Verificar que el archivo de video existe
    if not os.path.exists(args.video_path):
        print(f"❌ Error: No se encontró el archivo de video: {args.video_path}")
        return
    
    print(f"🎬 Procesando video: {args.video_path}")
    print(f"📁 Método: {args.method}")
    print(f"🎯 Umbral de calidad: {args.quality}")
    
    extractor = VideoFrameExtractor(
        video_path=args.video_path,
        output_dir=args.output,
        quality_threshold=args.quality
    )
    
    extractor.create_output_directory()
    
    try:
        if args.method == 'interval':
            extractor.extract_all_frames(interval_seconds=args.interval)
        elif args.method == 'scene':
            extractor.extract_scene_changes(min_interval_seconds=args.interval)
        elif args.method == 'adaptive':
            extractor.extract_adaptive(base_interval=args.interval)
        
        extractor.save_metadata()
        extractor.print_summary()
        
    except Exception as e:
        print(f"❌ Error durante la extracción: {str(e)}")
        return
    
    print("✅ Extracción completada exitosamente!")


if __name__ == "__main__":
    main()