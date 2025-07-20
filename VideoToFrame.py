import cv2
import os
import sys

def extraer_frames(ruta_video, carpeta_salida, intervalo=1):
    """
    Extrae frames de un video usando OpenCV
    
    Args:
        ruta_video (str): Ruta al archivo de video
        carpeta_salida (str): Carpeta donde guardar los frames
        intervalo (int): Extraer cada N frames (1 = todos los frames)
    """
    # Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Abrir el video
    cap = cv2.VideoCapture(ruta_video)
    
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video: {ruta_video}")
        return False
    
    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion = total_frames / fps
    
    print(f"FPS: {fps}")
    print(f"Total de frames: {total_frames}")
    print(f"Duración: {duracion:.2f} segundos")
    print(f"Extrayendo cada {intervalo} frame(s)...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        # Leer frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Guardar frame cada 'intervalo' frames
        if frame_count % intervalo == 0:
            nombre_frame = f"frame_{saved_count:06d}.jpg"
            ruta_frame = os.path.join(carpeta_salida, nombre_frame)
            cv2.imwrite(ruta_frame, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:  # Mostrar progreso
                print(f"Frames guardados: {saved_count}")
        
        frame_count += 1
    
    # Liberar recursos
    cap.release()
    print(f"Proceso completado. Total de frames guardados: {saved_count}")
    return True

def extraer_frames_rango(ruta_video, carpeta_salida, frame_inicio, frame_fin):
    """
    Extrae un rango específico de frames
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video: {ruta_video}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validar rango
    if frame_inicio < 0 or frame_fin >= total_frames or frame_inicio > frame_fin:
        print(f"Error: Rango inválido. El video tiene {total_frames} frames (0-{total_frames-1})")
        cap.release()
        return False
    
    print(f"Extrayendo frames del {frame_inicio} al {frame_fin}...")
    
    # Posicionar en el frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
    
    frame_actual = frame_inicio
    saved_count = 0
    
    while frame_actual <= frame_fin:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        nombre_frame = f"frame_{frame_actual:06d}.jpg"
        ruta_frame = os.path.join(carpeta_salida, nombre_frame)
        cv2.imwrite(ruta_frame, frame)
        saved_count += 1
        frame_actual += 1
    
    cap.release()
    print(f"Extraídos {saved_count} frames del rango {frame_inicio} a {frame_fin}")
    return True

def extraer_frames_tiempo(ruta_video, carpeta_salida, tiempo_inicio, tiempo_fin):
    """
    Extrae frames por rango de tiempo en segundos
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    cap = cv2.VideoCapture(ruta_video)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video: {ruta_video}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_total = total_frames / fps
    
    # Validar tiempos
    if tiempo_inicio < 0 or tiempo_fin > duracion_total or tiempo_inicio >= tiempo_fin:
        print(f"Error: Rango de tiempo inválido. El video dura {duracion_total:.2f} segundos")
        cap.release()
        return False
    
    frame_inicio = int(tiempo_inicio * fps)
    frame_fin = int(tiempo_fin * fps)
    
    print(f"Extrayendo frames de {tiempo_inicio}s a {tiempo_fin}s...")
    
    # Posicionar en el tiempo inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
    
    frame_actual = frame_inicio
    saved_count = 0
    
    while frame_actual <= frame_fin:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        tiempo_actual = frame_actual / fps
        nombre_frame = f"frame_{tiempo_actual:.2f}s.jpg"
        ruta_frame = os.path.join(carpeta_salida, nombre_frame)
        cv2.imwrite(ruta_frame, frame)
        saved_count += 1
        frame_actual += 1
    
    cap.release()
    print(f"Extraídos {saved_count} frames del tiempo {tiempo_inicio}s a {tiempo_fin}s")
    return True

def mostrar_ayuda():
    print("""
Uso: python extract_frames.py <video> <carpeta_salida> [opciones]

Argumentos:
  video           Ruta al archivo de video
  carpeta_salida  Carpeta donde guardar los frames

Opciones:
  -i <N>          Extraer cada N frames (ej: -i 30)
  -r <inicio> <fin>   Extraer rango de frames (ej: -r 100 500)
  -t <inicio> <fin>   Extraer rango de tiempo en segundos (ej: -t 10.0 20.0)
  -h              Mostrar esta ayuda

Ejemplos:
  python extract_frames.py video.mp4 frames/          # Todos los frames
  python extract_frames.py video.mp4 frames/ -i 30    # Cada 30 frames
  python extract_frames.py video.mp4 frames/ -r 100 500   # Frames 100-500
  python extract_frames.py video.mp4 frames/ -t 10.0 20.0 # De 10 a 20 segundos
    """)

if __name__ == "__main__":
    # Verificar argumentos mínimos
    if len(sys.argv) < 3 or '-h' in sys.argv:
        mostrar_ayuda()
        sys.exit(1)
    
    video = sys.argv[1]
    carpeta_salida = sys.argv[2]
    
    # Verificar que el archivo de video existe
    if not os.path.exists(video):
        print(f"Error: El archivo de video no existe: {video}")
        sys.exit(1)
    
    # Procesar argumentos opcionales
    if len(sys.argv) > 3:
        opcion = sys.argv[3]
        
        if opcion == '-i' and len(sys.argv) >= 5:
            # Extraer con intervalo
            intervalo = int(sys.argv[4])
            extraer_frames(video, carpeta_salida, intervalo)
            
        elif opcion == '-r' and len(sys.argv) >= 6:
            # Extraer rango de frames
            frame_inicio = int(sys.argv[4])
            frame_fin = int(sys.argv[5])
            extraer_frames_rango(video, carpeta_salida, frame_inicio, frame_fin)
            
        elif opcion == '-t' and len(sys.argv) >= 6:
            # Extraer rango de tiempo
            tiempo_inicio = float(sys.argv[4])
            tiempo_fin = float(sys.argv[5])
            extraer_frames_tiempo(video, carpeta_salida, tiempo_inicio, tiempo_fin)
            
        else:
            print("Error: Opción inválida o faltan argumentos")
            mostrar_ayuda()
            sys.exit(1)
    else:
        # Extraer todos los frames (por defecto)
        extraer_frames(video, carpeta_salida)