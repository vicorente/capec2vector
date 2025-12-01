"""
Tests para el módulo de configuración
"""
import unittest
import os
import sys
from pathlib import Path

# Añadir el directorio padre al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfig(unittest.TestCase):
    """Tests para la configuración del proyecto"""

    def test_project_root_exists(self):
        """Verifica que el directorio raíz del proyecto existe"""
        self.assertTrue(config.PROJECT_ROOT.exists())
        self.assertTrue(config.PROJECT_ROOT.is_dir())

    def test_capec_data_dir_exists(self):
        """Verifica que el directorio de datos CAPEC existe"""
        self.assertTrue(config.CAPEC_DATA_DIR.exists())
        self.assertTrue(config.CAPEC_DATA_DIR.is_dir())

    def test_xml_file_exists(self):
        """Verifica que el archivo XML de CAPEC existe"""
        self.assertTrue(config.XML_FILE_PATH.exists())
        self.assertTrue(config.XML_FILE_PATH.is_file())

    def test_milvus_config(self):
        """Verifica la configuración de Milvus"""
        self.assertIsInstance(config.MILVUS_HOST, str)
        self.assertIsInstance(config.MILVUS_PORT, int)
        self.assertGreater(config.MILVUS_PORT, 0)
        self.assertLess(config.MILVUS_PORT, 65536)
        self.assertIsInstance(config.COLLECTION_NAME, str)
        self.assertGreater(len(config.COLLECTION_NAME), 0)

    def test_embedding_config(self):
        """Verifica la configuración de embeddings"""
        self.assertIsInstance(config.EMBEDDING_MODEL, str)
        self.assertGreater(len(config.EMBEDDING_MODEL), 0)
        self.assertIsInstance(config.EMBEDDING_DIMENSION, int)
        self.assertGreater(config.EMBEDDING_DIMENSION, 0)

    def test_api_config(self):
        """Verifica la configuración de la API"""
        self.assertIsInstance(config.API_HOST, str)
        self.assertIsInstance(config.API_PORT, int)
        self.assertGreater(config.API_PORT, 0)
        self.assertLess(config.API_PORT, 65536)

    def test_search_config(self):
        """Verifica la configuración de búsqueda"""
        self.assertIsInstance(config.DEFAULT_SEARCH_TOP_K, int)
        self.assertGreater(config.DEFAULT_SEARCH_TOP_K, 0)
        self.assertIsInstance(config.MAX_SEARCH_TOP_K, int)
        self.assertGreater(config.MAX_SEARCH_TOP_K, config.DEFAULT_SEARCH_TOP_K)

    def test_kali_tools_config(self):
        """Verifica la configuración de herramientas Kali"""
        self.assertIsInstance(config.KALI_TOOLS, dict)
        self.assertGreater(len(config.KALI_TOOLS), 0)
        for tool, endpoint in config.KALI_TOOLS.items():
            self.assertIsInstance(tool, str)
            self.assertIsInstance(endpoint, str)
            self.assertTrue(endpoint.startswith("/"))

    def test_max_lengths_config(self):
        """Verifica la configuración de longitudes máximas"""
        self.assertIsInstance(config.MAX_LENGTHS, dict)
        self.assertGreater(len(config.MAX_LENGTHS), 0)
        for field, max_length in config.MAX_LENGTHS.items():
            self.assertIsInstance(field, str)
            self.assertIsInstance(max_length, int)
            self.assertGreater(max_length, 0)

    def test_get_config_summary(self):
        """Verifica que get_config_summary retorna un diccionario válido"""
        summary = config.get_config_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("milvus", summary)
        self.assertIn("embeddings", summary)
        self.assertIn("ollama", summary)
        self.assertIn("api", summary)
        self.assertIn("cache", summary)

    def test_cache_config(self):
        """Verifica la configuración de caché"""
        self.assertIsInstance(config.ENABLE_CACHE, bool)
        self.assertIsInstance(config.CACHE_TTL, int)
        self.assertGreater(config.CACHE_TTL, 0)


if __name__ == "__main__":
    unittest.main()
