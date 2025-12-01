"""
Validadores y utilidades para sanitización de datos
"""
import re
import html
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def sanitize_text(text: Optional[str], max_length: Optional[int] = None) -> str:
    """
    Limpia y sanitiza texto, eliminando caracteres problemáticos.

    Args:
        text: Texto a sanitizar
        max_length: Longitud máxima del texto (opcional)

    Returns:
        Texto sanitizado
    """
    if text is None:
        return ""

    # Convertir a string si no lo es
    text = str(text)

    # Eliminar espacios en blanco excesivos
    text = re.sub(r"\s+", " ", text)

    # Decodificar entidades HTML
    text = html.unescape(text)

    # Eliminar caracteres de control excepto saltos de línea y tabulaciones
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # Strip whitespace
    text = text.strip()

    # Truncar si se especifica max_length
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text


def validate_pattern_data(pattern_data: Dict[str, Any]) -> bool:
    """
    Valida que los datos de un patrón CAPEC sean correctos.

    Args:
        pattern_data: Diccionario con datos del patrón

    Returns:
        True si los datos son válidos, False en caso contrario
    """
    required_fields = ["pattern_id", "name"]

    # Verificar campos requeridos
    for field in required_fields:
        if field not in pattern_data:
            logger.error(f"Campo requerido ausente: {field}")
            return False
        if not pattern_data[field]:
            logger.error(f"Campo requerido vacío: {field}")
            return False

    # Validar pattern_id
    pattern_id = pattern_data.get("pattern_id", "")
    if not re.match(r"^\d+$", str(pattern_id)):
        logger.error(f"pattern_id inválido: {pattern_id}")
        return False

    # Validar name
    name = pattern_data.get("name", "")
    if len(name) < 3:
        logger.error(f"Nombre demasiado corto: {name}")
        return False

    return True


def validate_embedding(embedding: Any) -> bool:
    """
    Valida que un embedding sea correcto.

    Args:
        embedding: Embedding a validar

    Returns:
        True si el embedding es válido, False en caso contrario
    """
    try:
        import numpy as np

        # Convertir a numpy array si no lo es
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Verificar que sea un array 1D
        if embedding.ndim != 1:
            logger.error(f"Embedding tiene dimensión incorrecta: {embedding.ndim}")
            return False

        # Verificar que no contenga NaN o Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.error("Embedding contiene NaN o Inf")
            return False

        # Verificar que no esté vacío
        if embedding.size == 0:
            logger.error("Embedding está vacío")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validando embedding: {e}")
        return False


def truncate_field(text: str, max_length: int, field_name: str = "") -> str:
    """
    Trunca un campo de texto a la longitud máxima permitida.

    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        field_name: Nombre del campo (para logging)

    Returns:
        Texto truncado
    """
    if not text:
        return ""

    if len(text) > max_length:
        logger.warning(
            f"Campo {field_name} truncado de {len(text)} a {max_length} caracteres"
        )
        return text[:max_length]

    return text


def validate_collection_schema(fields: list) -> bool:
    """
    Valida que el esquema de una colección sea correcto.

    Args:
        fields: Lista de campos del esquema

    Returns:
        True si el esquema es válido, False en caso contrario
    """
    if not fields:
        logger.error("Esquema vacío")
        return False

    # Verificar que haya al menos un campo de embedding
    has_vector_field = any(
        hasattr(field, "dtype") and "VECTOR" in str(field.dtype) for field in fields
    )

    if not has_vector_field:
        logger.error("Esquema no contiene campo vectorial")
        return False

    return True
