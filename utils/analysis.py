"""
Herramientas de análisis y estadísticas para patrones CAPEC
"""
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import logging
from pymilvus import connections, Collection
import numpy as np

logger = logging.getLogger(__name__)


class CAPECAnalyzer:
    """Clase para analizar patrones CAPEC almacenados en Milvus"""

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 19530):
        """
        Inicializa el analizador.

        Args:
            collection_name: Nombre de la colección de Milvus
            host: Host de Milvus
            port: Puerto de Milvus
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None

    def connect(self):
        """Conecta a Milvus y carga la colección"""
        try:
            connections.connect(host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"Conectado a colección {self.collection_name}")
        except Exception as e:
            logger.error(f"Error conectando a Milvus: {e}")
            raise

    def disconnect(self):
        """Desconecta de Milvus"""
        try:
            connections.disconnect("default")
        except Exception as e:
            logger.warning(f"Error desconectando: {e}")

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas básicas de la colección.

        Returns:
            Diccionario con estadísticas
        """
        try:
            self.connect()

            stats = {
                "total_patterns": self.collection.num_entities,
                "collection_name": self.collection_name,
            }

            # Obtener todos los patrones para análisis
            results = self.collection.query(
                expr="pattern_id != ''",
                output_fields=["pattern_id", "name", "status", "abstraction"],
                limit=10000,
            )

            if results:
                # Analizar status
                status_counts = Counter(r.get("status", "Unknown") for r in results)
                stats["by_status"] = dict(status_counts)

                # Analizar abstraction
                abstraction_counts = Counter(
                    r.get("abstraction", "Unknown") for r in results
                )
                stats["by_abstraction"] = dict(abstraction_counts)

            return stats

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            raise
        finally:
            self.disconnect()

    def get_severity_distribution(self) -> Dict[str, int]:
        """
        Obtiene la distribución de severidades.

        Returns:
            Diccionario con conteo por severidad
        """
        try:
            self.connect()

            results = self.collection.query(
                expr="pattern_id != ''",
                output_fields=["typical_severity"],
                limit=10000,
            )

            severity_counts = Counter(
                r.get("typical_severity", "Not Specified") for r in results
            )

            return dict(severity_counts)

        except Exception as e:
            logger.error(f"Error obteniendo distribución de severidad: {e}")
            raise
        finally:
            self.disconnect()

    def get_likelihood_distribution(self) -> Dict[str, int]:
        """
        Obtiene la distribución de probabilidades de ataque.

        Returns:
            Diccionario con conteo por probabilidad
        """
        try:
            self.connect()

            results = self.collection.query(
                expr="pattern_id != ''",
                output_fields=["likelihood_of_attack"],
                limit=10000,
            )

            likelihood_counts = Counter(
                r.get("likelihood_of_attack", "Not Specified") for r in results
            )

            return dict(likelihood_counts)

        except Exception as e:
            logger.error(f"Error obteniendo distribución de probabilidad: {e}")
            raise
        finally:
            self.disconnect()

    def get_top_patterns_by_field(
        self, field: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtiene los patrones más frecuentes según un campo.

        Args:
            field: Campo a analizar
            top_n: Número de resultados a retornar

        Returns:
            Lista de patrones ordenados por frecuencia
        """
        try:
            self.connect()

            results = self.collection.query(
                expr="pattern_id != ''",
                output_fields=["pattern_id", "name", field],
                limit=10000,
            )

            # Contar frecuencias
            field_counts = Counter(r.get(field, "Unknown") for r in results)

            # Obtener top N
            top_items = field_counts.most_common(top_n)

            return [{"value": item, "count": count} for item, count in top_items]

        except Exception as e:
            logger.error(f"Error obteniendo top patterns: {e}")
            raise
        finally:
            self.disconnect()

    def search_patterns_by_keyword(
        self, keyword: str, fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca patrones que contengan una palabra clave.

        Args:
            keyword: Palabra clave a buscar
            fields: Campos en los que buscar (por defecto: name, description)

        Returns:
            Lista de patrones que coinciden
        """
        if fields is None:
            fields = ["pattern_id", "name", "description"]

        try:
            self.connect()

            # Construir expresión de búsqueda
            # Nota: Milvus tiene limitaciones en búsqueda de texto
            # Esta es una búsqueda básica que obtiene todos y filtra
            results = self.collection.query(
                expr="pattern_id != ''",
                output_fields=fields,
                limit=10000,
            )

            # Filtrar por keyword
            keyword_lower = keyword.lower()
            matching_patterns = []

            for result in results:
                for field in fields:
                    field_value = str(result.get(field, ""))
                    if keyword_lower in field_value.lower():
                        matching_patterns.append(result)
                        break

            logger.info(f"Encontrados {len(matching_patterns)} patrones con '{keyword}'")
            return matching_patterns

        except Exception as e:
            logger.error(f"Error buscando patrones: {e}")
            raise
        finally:
            self.disconnect()

    def generate_report(self, output_file: str = None) -> str:
        """
        Genera un reporte completo de análisis.

        Args:
            output_file: Archivo donde guardar el reporte (opcional)

        Returns:
            Reporte en formato texto
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("REPORTE DE ANÁLISIS CAPEC")
            report_lines.append("=" * 80)
            report_lines.append("")

            # Estadísticas básicas
            basic_stats = self.get_basic_stats()
            report_lines.append("ESTADÍSTICAS BÁSICAS")
            report_lines.append("-" * 80)
            report_lines.append(f"Total de patrones: {basic_stats['total_patterns']}")
            report_lines.append("")

            if "by_status" in basic_stats:
                report_lines.append("Distribución por Status:")
                for status, count in basic_stats["by_status"].items():
                    percentage = (count / basic_stats["total_patterns"]) * 100
                    report_lines.append(f"  {status}: {count} ({percentage:.1f}%)")
                report_lines.append("")

            if "by_abstraction" in basic_stats:
                report_lines.append("Distribución por Abstraction:")
                for abstraction, count in basic_stats["by_abstraction"].items():
                    percentage = (count / basic_stats["total_patterns"]) * 100
                    report_lines.append(
                        f"  {abstraction}: {count} ({percentage:.1f}%)"
                    )
                report_lines.append("")

            # Distribución de severidad
            severity_dist = self.get_severity_distribution()
            report_lines.append("DISTRIBUCIÓN DE SEVERIDAD")
            report_lines.append("-" * 80)
            for severity, count in sorted(
                severity_dist.items(), key=lambda x: x[1], reverse=True
            ):
                report_lines.append(f"{severity}: {count}")
            report_lines.append("")

            # Distribución de probabilidad
            likelihood_dist = self.get_likelihood_distribution()
            report_lines.append("DISTRIBUCIÓN DE PROBABILIDAD DE ATAQUE")
            report_lines.append("-" * 80)
            for likelihood, count in sorted(
                likelihood_dist.items(), key=lambda x: x[1], reverse=True
            ):
                report_lines.append(f"{likelihood}: {count}")
            report_lines.append("")

            report_lines.append("=" * 80)

            report = "\n".join(report_lines)

            # Guardar a archivo si se especifica
            if output_file:
                with open(output_file, "w") as f:
                    f.write(report)
                logger.info(f"Reporte guardado en {output_file}")

            return report

        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            raise


def main():
    """Función principal para uso desde línea de comandos"""
    import argparse

    parser = argparse.ArgumentParser(description="Análisis de patrones CAPEC")
    parser.add_argument(
        "--collection",
        default="capec_patterns",
        help="Nombre de la colección de Milvus",
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
        "--output",
        help="Archivo de salida para el reporte",
    )
    parser.add_argument(
        "--search",
        help="Buscar patrones con palabra clave",
    )

    args = parser.parse_args()

    analyzer = CAPECAnalyzer(args.collection, args.host, args.port)

    if args.search:
        results = analyzer.search_patterns_by_keyword(args.search)
        print(f"\nEncontrados {len(results)} patrones:")
        for i, pattern in enumerate(results[:10], 1):
            print(f"{i}. {pattern.get('pattern_id')}: {pattern.get('name')}")
    else:
        report = analyzer.generate_report(args.output)
        print(report)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
