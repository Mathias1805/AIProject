from fuzzywuzzy import fuzz
from Levenshtein import distance  # requiere python-Levenshtein

def comparar_similitud_precisa(texto1, texto2):
    """
    Compara dos cadenas de texto y devuelve un porcentaje de similitud como float.
    Usa la distancia de Levenshtein normalizada para obtener más precisión.

    Retorna un valor entre 0.00 y 100.00
    """
    max_len = max(len(texto1), len(texto2))
    if max_len == 0:
        return 100.0  # ambas vacías = 100% similares
    dist = distance(texto1, texto2)
    similitud = (1 - dist / max_len) * 100
    return round(similitud, 2)

# Ejemplo de uso
resultado = comparar_similitud_precisa("", "")
print(f"Similitud: {resultado}%")
