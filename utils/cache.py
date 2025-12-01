"""
Sistema de caché para embeddings generados.
Evita regenerar embeddings para textos que ya han sido procesados.
"""
import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional, List
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Caché para almacenar y recuperar embeddings.
    Usa el hash del texto como clave para búsquedas rápidas.
    """

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        """
        Inicializa el caché de embeddings.

        Args:
            cache_dir: Directorio donde se almacenarán los embeddings
            ttl_seconds: Tiempo de vida de los elementos en caché (segundos)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Carga el metadata del caché"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando metadata del caché: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Guarda el metadata del caché"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata del caché: {e}")

    def _get_hash(self, text: str) -> str:
        """Genera un hash SHA256 del texto"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cache_path(self, text_hash: str) -> Path:
        """Obtiene la ruta del archivo de caché para un hash dado"""
        return self.cache_dir / f"{text_hash}.pkl"

    def _is_expired(self, timestamp: str) -> bool:
        """Verifica si un elemento del caché ha expirado"""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            expiry_time = cached_time + timedelta(seconds=self.ttl_seconds)
            return datetime.now() > expiry_time
        except Exception:
            return True

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Recupera un embedding del caché.

        Args:
            text: Texto del cual recuperar el embedding

        Returns:
            El embedding si existe y no ha expirado, None en caso contrario
        """
        text_hash = self._get_hash(text)
        cache_path = self._get_cache_path(text_hash)

        # Verificar si existe en metadata
        if text_hash not in self.metadata:
            return None

        # Verificar si ha expirado
        if self._is_expired(self.metadata[text_hash]["timestamp"]):
            logger.debug(f"Embedding expirado para hash {text_hash[:8]}...")
            self.delete(text)
            return None

        # Intentar cargar el embedding
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                logger.debug(f"Embedding recuperado del caché para hash {text_hash[:8]}...")
                return embedding
            except Exception as e:
                logger.error(f"Error cargando embedding del caché: {e}")
                self.delete(text)
                return None

        return None

    def set(self, text: str, embedding: np.ndarray):
        """
        Almacena un embedding en el caché.

        Args:
            text: Texto para el cual almacenar el embedding
            embedding: El embedding a almacenar
        """
        text_hash = self._get_hash(text)
        cache_path = self._get_cache_path(text_hash)

        try:
            # Guardar el embedding
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)

            # Actualizar metadata
            self.metadata[text_hash] = {
                "timestamp": datetime.now().isoformat(),
                "text_preview": text[:100] if len(text) > 100 else text,
                "embedding_shape": embedding.shape,
            }
            self._save_metadata()
            logger.debug(f"Embedding almacenado en caché para hash {text_hash[:8]}...")

        except Exception as e:
            logger.error(f"Error guardando embedding en caché: {e}")

    def delete(self, text: str):
        """
        Elimina un embedding del caché.

        Args:
            text: Texto cuyo embedding eliminar
        """
        text_hash = self._get_hash(text)
        cache_path = self._get_cache_path(text_hash)

        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception as e:
                logger.error(f"Error eliminando archivo de caché: {e}")

        if text_hash in self.metadata:
            del self.metadata[text_hash]
            self._save_metadata()

    def clear(self):
        """Elimina todos los elementos del caché"""
        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            self.metadata = {}
            self._save_metadata()
            logger.info("Caché limpiado completamente")
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")

    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del caché.

        Returns:
            Diccionario con estadísticas del caché
        
        Nota: Para grandes volúmenes de caché, considera implementar
        contadores incrementales en lugar de recalcular en cada llamada.
        """
        total_items = len(self.metadata)
        # Nota: Para cachés grandes, esta iteración puede ser costosa
        # Considera mantener un contador de items expirados si el rendimiento es crítico
        expired_items = sum(
            1 for item in self.metadata.values() if self._is_expired(item["timestamp"])
        )
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "total_items": total_items,
            "active_items": total_items - expired_items,
            "expired_items": expired_items,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
