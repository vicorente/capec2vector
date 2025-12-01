#!/usr/bin/env python3
"""
Ejemplo de uso de las herramientas de análisis de patrones CAPEC
"""
import sys
from pathlib import Path

# Añadir el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Nota: Este ejemplo requiere que Milvus esté corriendo y la colección cargada
# Si no tienes Milvus disponible, este ejemplo mostrará errores de conexión


def example_connection_check():
    """Verifica la conexión antes de ejecutar ejemplos"""
    print("\n" + "="*80)
    print("Verificando conexión a Milvus...")
    print("="*80 + "\n")
    
    try:
        from pymilvus import connections
        connections.connect(host="localhost", port=19530)
        print("✅ Conectado a Milvus exitosamente")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"❌ No se puede conectar a Milvus: {e}")
        print("\n⚠️  Para usar este ejemplo necesitas:")
        print("  1. Docker corriendo: docker-compose up -d")
        print("  2. Colección cargada: python embeddings.py")
        return False


def example_basic_stats():
    """Ejemplo de estadísticas básicas"""
    from utils.analysis import CAPECAnalyzer
    
    print("\n" + "="*80)
    print("Ejemplo 1: Estadísticas básicas")
    print("="*80 + "\n")
    
    analyzer = CAPECAnalyzer("capec_patterns")
    
    try:
        stats = analyzer.get_basic_stats()
        
        print(f"📊 Total de patrones: {stats['total_patterns']}")
        print()
        
        if "by_status" in stats:
            print("Distribución por Status:")
            for status, count in stats["by_status"].items():
                percentage = (count / stats["total_patterns"]) * 100
                print(f"  {status}: {count} ({percentage:.1f}%)")
        
        print()
        
        if "by_abstraction" in stats:
            print("Distribución por Abstraction:")
            for abstraction, count in stats["by_abstraction"].items():
                percentage = (count / stats["total_patterns"]) * 100
                print(f"  {abstraction}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_severity_analysis():
    """Ejemplo de análisis de severidad"""
    from utils.analysis import CAPECAnalyzer
    
    print("\n" + "="*80)
    print("Ejemplo 2: Análisis de severidad")
    print("="*80 + "\n")
    
    analyzer = CAPECAnalyzer("capec_patterns")
    
    try:
        severity_dist = analyzer.get_severity_distribution()
        
        print("⚠️  Distribución de Severidad:\n")
        
        # Ordenar por severidad (más severo primero)
        severity_order = ["Very High", "High", "Medium", "Low", "Very Low", "Not Specified"]
        
        for severity in severity_order:
            if severity in severity_dist:
                count = severity_dist[severity]
                print(f"  {severity:15} : {count:3} patrones")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_likelihood_analysis():
    """Ejemplo de análisis de probabilidad"""
    from utils.analysis import CAPECAnalyzer
    
    print("\n" + "="*80)
    print("Ejemplo 3: Análisis de probabilidad de ataque")
    print("="*80 + "\n")
    
    analyzer = CAPECAnalyzer("capec_patterns")
    
    try:
        likelihood_dist = analyzer.get_likelihood_distribution()
        
        print("🎯 Distribución de Probabilidad de Ataque:\n")
        
        # Ordenar por probabilidad
        likelihood_order = ["Very High", "High", "Medium", "Low", "Very Low", "Not Specified"]
        
        for likelihood in likelihood_order:
            if likelihood in likelihood_dist:
                count = likelihood_dist[likelihood]
                print(f"  {likelihood:15} : {count:3} patrones")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_keyword_search():
    """Ejemplo de búsqueda por palabra clave"""
    from utils.analysis import CAPECAnalyzer
    
    print("\n" + "="*80)
    print("Ejemplo 4: Búsqueda por palabra clave")
    print("="*80 + "\n")
    
    analyzer = CAPECAnalyzer("capec_patterns")
    
    keywords = ["injection", "overflow", "scripting"]
    
    for keyword in keywords:
        try:
            print(f"🔍 Buscando patrones con '{keyword}'...")
            results = analyzer.search_patterns_by_keyword(keyword)
            
            print(f"  Encontrados: {len(results)} patrones")
            
            # Mostrar los primeros 3
            for i, pattern in enumerate(results[:3], 1):
                print(f"  {i}. {pattern.get('pattern_id')}: {pattern.get('name')}")
            
            if len(results) > 3:
                print(f"  ... y {len(results) - 3} más")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")


def example_generate_report():
    """Ejemplo de generación de reporte completo"""
    from utils.analysis import CAPECAnalyzer
    
    print("\n" + "="*80)
    print("Ejemplo 5: Generación de reporte completo")
    print("="*80 + "\n")
    
    analyzer = CAPECAnalyzer("capec_patterns")
    
    try:
        print("📝 Generando reporte completo...")
        report = analyzer.generate_report()
        
        # Mostrar solo las primeras líneas
        lines = report.split("\n")
        print("\n".join(lines[:30]))
        print("...")
        print(f"\n(Reporte completo tiene {len(lines)} líneas)")
        
        # Guardar a archivo
        output_file = "capec_analysis_report.txt"
        with open(output_file, "w") as f:
            f.write(report)
        
        print(f"\n✅ Reporte guardado en: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Ejecuta todos los ejemplos"""
    print("\n🎯 Ejemplos de análisis de patrones CAPEC\n")
    
    # Verificar conexión primero
    if not example_connection_check():
        return 1
    
    try:
        example_basic_stats()
        input("\nPresiona Enter para continuar...")
        
        example_severity_analysis()
        input("\nPresiona Enter para continuar...")
        
        example_likelihood_analysis()
        input("\nPresiona Enter para continuar...")
        
        example_keyword_search()
        input("\nPresiona Enter para continuar...")
        
        example_generate_report()
        
        print("\n" + "="*80)
        print("✅ Todos los ejemplos completados exitosamente!")
        print("="*80)
        
        print("\n💡 Tips:")
        print("  - Usa analysis.py desde línea de comandos para análisis rápidos")
        print("  - Genera reportes periódicos para monitorear tu base de datos")
        print("  - Combina búsquedas por keyword con análisis estadísticos")
        print("  - Exporta reportes para documentación y auditorías")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
