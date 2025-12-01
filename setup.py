#!/usr/bin/env python3
"""
Script de inicialización para el proyecto capec2vector.
Verifica dependencias, crea directorios necesarios y valida la configuración.
"""
import sys
import os
import subprocess
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Verifica que la versión de Python sea >= 3.8"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ requerido. Versión actual: {sys.version}")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    required_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "pymilvus",
        "fastapi",
        "uvicorn",
        "numpy",
        "requests",
        "ollama",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package} no instalado")
            missing_packages.append(package)

    if missing_packages:
        logger.warning(
            f"\nPaquetes faltantes: {', '.join(missing_packages)}"
        )
        logger.info("Ejecuta: pip install -r requirements.txt")
        return False

    return True


def create_directories():
    """Crea directorios necesarios para el proyecto"""
    directories = [
        ".cache",
        "logs",
        "exports",
        "backups",
    ]

    project_root = Path(__file__).parent
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"✓ Directorio creado/verificado: {directory}")

    return True


def check_capec_data():
    """Verifica que los datos CAPEC estén disponibles"""
    project_root = Path(__file__).parent
    xml_file = project_root / "capec_latest" / "capec_v3.9.xml"

    if not xml_file.exists():
        logger.error(f"✗ Archivo CAPEC no encontrado: {xml_file}")
        logger.info("Descarga el archivo desde: https://capec.mitre.org/data/xml/capec_latest.xml")
        return False

    logger.info(f"✓ Archivo CAPEC encontrado: {xml_file}")
    file_size = xml_file.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"  Tamaño: {file_size:.2f} MB")
    return True


def check_docker():
    """Verifica que Docker esté instalado y funcionando"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"✓ Docker: {result.stdout.strip()}")
            return True
        else:
            logger.warning("✗ Docker no está funcionando correctamente")
            return False
    except FileNotFoundError:
        logger.warning("✗ Docker no está instalado")
        logger.info("Instala Docker desde: https://docs.docker.com/get-docker/")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("✗ Docker no responde")
        return False


def check_milvus():
    """Verifica si Milvus está corriendo"""
    try:
        from pymilvus import connections

        connections.connect(host="localhost", port=19530)
        logger.info("✓ Milvus está corriendo y accesible")
        connections.disconnect("default")
        return True
    except Exception as e:
        logger.warning(f"✗ No se puede conectar a Milvus: {e}")
        logger.info("Inicia Milvus con: docker-compose up -d")
        return False


def check_ollama():
    """Verifica si Ollama está disponible"""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"✓ Ollama está corriendo ({len(models)} modelos disponibles)")
            return True
        else:
            logger.warning("✗ Ollama no responde correctamente")
            return False
    except Exception as e:
        logger.warning(f"✗ No se puede conectar a Ollama: {e}")
        logger.info("Instala Ollama desde: https://ollama.ai/")
        return False


def print_summary(checks):
    """Imprime resumen de verificaciones"""
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE VERIFICACIÓN")
    logger.info("=" * 60)

    passed = sum(checks.values())
    total = len(checks)

    for check, result in checks.items():
        status = "✓" if result else "✗"
        logger.info(f"{status} {check}")

    logger.info(f"\nResultado: {passed}/{total} verificaciones pasadas")

    if passed == total:
        logger.info("\n✓ ¡Sistema listo para usar!")
        logger.info("\nPróximos pasos:")
        logger.info("1. Inicia Milvus: docker-compose up -d")
        logger.info("2. Genera embeddings: python embeddings.py")
        logger.info("3. Inicia la API: uvicorn ollama_milvus_bridge:app --reload")
    else:
        logger.info("\n⚠ Hay componentes que requieren atención")


def main():
    """Función principal"""
    logger.info("=" * 60)
    logger.info("VERIFICACIÓN DE SETUP - capec2vector")
    logger.info("=" * 60 + "\n")

    checks = {
        "Versión de Python": check_python_version(),
        "Dependencias Python": check_dependencies(),
        "Directorios del proyecto": create_directories(),
        "Datos CAPEC": check_capec_data(),
        "Docker": check_docker(),
    }

    # Verificaciones opcionales (no críticas)
    checks["Milvus"] = check_milvus()
    checks["Ollama"] = check_ollama()

    print_summary(checks)

    # Retornar código de salida
    critical_checks = ["Versión de Python", "Directorios del proyecto", "Datos CAPEC"]
    if all(checks[check] for check in critical_checks):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
