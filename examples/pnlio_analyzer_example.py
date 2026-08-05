# ejemplo de uso del analizador PNLIO (guarda gráfica automáticamente y resultados)
# archivo: examples/pnlio_analyzer_example.py

from pnlio import PNLIO_Coherence_Analyzer
import matplotlib.pyplot as plt
import json
import os

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
results = analyzer.print_report(dialogos, model_name='all-MiniLM-L6-v2')

# Crear carpeta de salida
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

# Guardar resultados como JSON si están disponibles
if results is not None:
    try:
        with open(os.path.join(out_dir, 'pnlio_report.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados: {os.path.join(out_dir, 'pnlio_report.json')}")
    except Exception as e:
        print("No se pudo guardar JSON de resultados:", e)

# Intentar guardar la figura en PNG. Dependiendo de la implementación de PNLIO, la figura
# puede venir dentro de `results['figure']`, `results.get('fig')`, o bien el analizador
# puede haber usado matplotlib directamente (capturable con plt).
saved = False
# Caso 1: figura dentro de results
if isinstance(results, dict):
    for key in ('figure', 'fig', 'plot'):
        if key in results:
            fig = results[key]
            try:
                fig_path = os.path.join(out_dir, 'pnlio_report.png')
                fig.savefig(fig_path, bbox_inches='tight')
                print(f"Gráfica guardada desde results['{key}']: {fig_path}")
                saved = True
                break
            except Exception:
                pass

# Caso 2: intentar guardar la figura actual de matplotlib
if not saved:
    try:
        fig_path = os.path.join(out_dir, 'pnlio_report.png')
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Gráfica guardada desde matplotlib: {fig_path}")
        saved = True
    except Exception as e:
        print("No se pudo guardar la gráfica automáticamente:", e)

if not saved:
    print("No se detectó una gráfica para guardar. Revisa cómo PNLIO genera figuras en tu versión del paquete.")
