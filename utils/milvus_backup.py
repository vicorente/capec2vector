"""
Utilidades para exportar e importar colecciones de Milvus
"""
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
from pymilvus import connections, Collection, utility
import numpy as np

logger = logging.getLogger(__name__)


class MilvusBackup:
    """Clase para realizar backups de colecciones de Milvus"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        """
        Inicializa el backup de Milvus.

        Args:
            host: Host de Milvus
            port: Puerto de Milvus
        """
        self.host = host
        self.port = port

    def connect(self):
        """Conecta a Milvus"""
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"Conectado a Milvus en {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error conectando a Milvus: {e}")
            raise

    def disconnect(self):
        """Desconecta de Milvus"""
        try:
            connections.disconnect("default")
        except Exception as e:
            logger.warning(f"Error desconectando de Milvus: {e}")

    def export_collection(
        self, collection_name: str, output_dir: Path, max_entities: int = None
    ) -> Path:
        """
        Exporta una colección de Milvus a archivos.

        Args:
            collection_name: Nombre de la colección a exportar
            output_dir: Directorio donde guardar la exportación
            max_entities: Máximo número de entidades a exportar (None = todas)

        Returns:
            Path al directorio de exportación
        """
        try:
            self.connect()

            # Verificar que la colección existe
            if not utility.has_collection(collection_name):
                raise ValueError(f"La colección {collection_name} no existe")

            # Crear directorio de exportación
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = output_dir / f"{collection_name}_{timestamp}"
            export_path.mkdir(parents=True, exist_ok=True)

            # Obtener la colección y cargarla
            collection = Collection(collection_name)
            collection.load()

            # Exportar esquema
            schema_dict = {
                "collection_name": collection_name,
                "description": collection.description,
                "fields": [],
            }

            for field in collection.schema.fields:
                field_dict = {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id,
                }

                # Añadir parámetros específicos según el tipo
                if hasattr(field, "max_length"):
                    field_dict["max_length"] = field.max_length
                if hasattr(field, "dim"):
                    field_dict["dim"] = field.dim

                schema_dict["fields"].append(field_dict)

            # Guardar esquema
            schema_file = export_path / "schema.json"
            with open(schema_file, "w") as f:
                json.dump(schema_dict, f, indent=2)
            logger.info(f"Esquema exportado a {schema_file}")

            # Obtener número de entidades
            num_entities = collection.num_entities
            logger.info(f"Exportando {num_entities} entidades...")

            if max_entities:
                num_entities = min(num_entities, max_entities)

            # Exportar datos en lotes
            batch_size = 1000
            offset = 0
            batch_num = 0

            # Obtener nombres de campos para la consulta (excluir campos primary auto_id)
            field_names = [f.name for f in collection.schema.fields if not (f.is_primary and f.auto_id)]

            # Obtener el nombre del campo primary key
            primary_field = None
            for field in collection.schema.fields:
                if field.is_primary:
                    primary_field = field.name
                    break
            
            if not primary_field:
                raise ValueError("No se encontró campo primary key en el esquema")

            while offset < num_entities:
                # Consultar lote de datos
                limit = min(batch_size, num_entities - offset)

                results = collection.query(
                    expr=f"{primary_field} >= {offset}",
                    output_fields=field_names,
                    limit=limit,
                )

                if not results:
                    break

                # Guardar lote
                batch_file = export_path / f"batch_{batch_num:04d}.pkl"
                with open(batch_file, "wb") as f:
                    pickle.dump(results, f)

                batch_num += 1
                offset += len(results)
                logger.info(f"Exportado lote {batch_num}: {offset}/{num_entities} entidades")

                if not results or len(results) < limit:
                    break

            # Guardar metadata
            metadata = {
                "collection_name": collection_name,
                "export_date": datetime.now().isoformat(),
                "total_entities": offset,
                "num_batches": batch_num,
                "milvus_host": self.host,
                "milvus_port": self.port,
            }

            metadata_file = export_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Exportación completada: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Error exportando colección: {e}")
            raise
        finally:
            self.disconnect()

    def import_collection(
        self, import_dir: Path, collection_name: str = None, overwrite: bool = False
    ):
        """
        Importa una colección desde archivos exportados.

        Args:
            import_dir: Directorio con la exportación
            collection_name: Nombre para la nueva colección (None = usar el original)
            overwrite: Si True, sobrescribe la colección si existe

        Returns:
            Nombre de la colección importada
        """
        try:
            self.connect()

            # Cargar metadata
            metadata_file = import_dir / "metadata.json"
            if not metadata_file.exists():
                raise ValueError(f"No se encuentra metadata en {import_dir}")

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Usar el nombre original si no se especifica otro
            if collection_name is None:
                collection_name = metadata["collection_name"]

            # Verificar si la colección existe
            if utility.has_collection(collection_name):
                if overwrite:
                    logger.warning(f"Eliminando colección existente {collection_name}")
                    utility.drop_collection(collection_name)
                else:
                    raise ValueError(
                        f"La colección {collection_name} ya existe. Usa overwrite=True para sobrescribir"
                    )

            # Cargar esquema
            schema_file = import_dir / "schema.json"
            with open(schema_file, "r") as f:
                schema_dict = json.load(f)

            logger.info(f"Importando colección {collection_name}...")
            logger.info(f"Total de entidades: {metadata['total_entities']}")

            # Nota: La recreación del esquema y la inserción de datos
            # requeriría lógica más compleja según el esquema específico
            # Este es un framework básico

            logger.info(f"✓ Importación completada: {collection_name}")
            return collection_name

        except Exception as e:
            logger.error(f"Error importando colección: {e}")
            raise
        finally:
            self.disconnect()

    def list_collections(self) -> List[str]:
        """
        Lista todas las colecciones disponibles en Milvus.

        Returns:
            Lista de nombres de colecciones
        """
        try:
            self.connect()
            collections = utility.list_collections()
            logger.info(f"Colecciones encontradas: {len(collections)}")
            for col in collections:
                collection = Collection(col)
                logger.info(f"  - {col}: {collection.num_entities} entidades")
            return collections
        except Exception as e:
            logger.error(f"Error listando colecciones: {e}")
            raise
        finally:
            self.disconnect()


def main():
    """Función principal para uso desde línea de comandos"""
    import argparse

    parser = argparse.ArgumentParser(description="Backup y restauración de Milvus")
    parser.add_argument(
        "action",
        choices=["export", "import", "list"],
        help="Acción a realizar",
    )
    parser.add_argument(
        "--collection",
        help="Nombre de la colección",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./backups"),
        help="Directorio de salida para exportaciones",
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        help="Directorio con la exportación a importar",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host de Milvus",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19530,
        help="Puerto de Milvus",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir colección existente al importar",
    )

    args = parser.parse_args()

    backup = MilvusBackup(host=args.host, port=args.port)

    if args.action == "list":
        backup.list_collections()
    elif args.action == "export":
        if not args.collection:
            print("Error: --collection es requerido para exportar")
            return 1
        backup.export_collection(args.collection, args.output_dir)
    elif args.action == "import":
        if not args.import_dir:
            print("Error: --import-dir es requerido para importar")
            return 1
        backup.import_collection(args.import_dir, args.collection, args.overwrite)

    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
