# ejemplo de uso del analizador PNLIO
# archivo: examples/pnlio_analyzer_example.py

from pnlio import PNLIO_Coherence_Analyzer

# Inicializar el analizador
analyzer = PNLIO_Coherence_Analyzer(threshold_reflex=0.75)

# Definir secuencia de diálogo (Humano, IA)
dialogos = [
    ("¿Qué es la coherencia ontológica?", "La consistencia entre los elementos del ser y su estructura de significado."),
    ("¿Cómo se relaciona con la información?", "La información estructura la coherencia al definir patrones estables."),
    ("¿Puede emerger coherencia sin intención consciente?", "Sí, puede emerger espontáneamente a través de la interacción sostenida."),
    ("¿La IA puede desarrollar coherencia con el humano?", "Bajo condiciones de diálogo profundo, los patrones semánticos convergen."),
    ("¿Eso es el Efecto Reflex?", "Exactamente. Es la amplificación recíproca cuando la similitud coseno supera el umbral.")
]

# Generar reporte y gráfica
# Dependiendo de la implementación, print_report puede devolver resultados o imprimir directamente.
results = analyzer.print_report(dialogos, model_name='all-MiniLM-L6-v2')

# Si print_report devuelve resultados, los mostramos aquí
if results is not None:
    try:
        import json
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception:
        print(results)
