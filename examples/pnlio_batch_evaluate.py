#!/usr/bin/env python3
# script para ejecutar evaluaciones en batch con distintos umbrales y modelos
# archivo: examples/pnlio_batch_evaluate.py

import os
import json
import csv
from pnlio import PNLIO_Coherence_Analyzer

# Diálogo de ejemplo (reutilizamos el mismo que en el ejemplo simple)
dialogos = [
    ("¿Qué es la coherencia ontológica?", "La consistencia entre los elementos del ser y su estructura de significado."),
    ("¿Cómo se relaciona con la información?", "La información estructura la coherencia al definir patrones estables."),
    ("¿Puede emerger coherencia sin intención consciente?", "Sí, puede emerger espontáneamente a través de la interacción sostenida."),
    ("¿La IA puede desarrollar coherencia con el humano?", "Bajo condiciones de diálogo profundo, los patrones semánticos convergen."),
    ("¿Eso es el Efecto Reflex?", "Exactamente. Es la amplificación recíproca cuando la similitud coseno supera el umbral.")
]

# Configuración por defecto (puedes editar o pasar parámetros más adelante)
models = [
    'all-MiniLM-L6-v2',
    'paraphrase-MiniLM-L6-v2'
]
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8]

out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)

results_list = []

for model in models:
    for thr in thresholds:
        print(f"Ejecutando modelo={model} umbral={thr} ...")
        analyzer = PNLIO_Coherence_Analyzer(threshold_reflex=thr)
        try:
            res = analyzer.print_report(dialogos, model_name=model)
        except Exception as e:
            res = {'error': str(e)}
        entry = {
            'model': model,
            'threshold_reflex': thr,
            'raw_results': res
        }
        # Intentar extraer métricas numéricas resumen para CSV (si existen)
        summary_metrics = {}
        if isinstance(res, dict):
            # ejemplos de claves posibles: 'metrics', 'summary', 'scores'
            for k in ('metrics', 'summary', 'scores'):
                if k in res and isinstance(res[k], dict):
                    summary_metrics.update(res[k])
        entry['summary_metrics'] = summary_metrics
        results_list.append(entry)

# Guardar JSON con todo
json_path = os.path.join(out_dir, 'batch_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_list, f, ensure_ascii=False, indent=2)
print(f"Resultados batch guardados en: {json_path}")

# Guardar CSV con columnas: model, threshold, summary_metrics (serializado)
csv_path = os.path.join(out_dir, 'batch_results.csv')
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'threshold_reflex', 'summary_metrics_json'])
    for r in results_list:
        writer.writerow([r['model'], r['threshold_reflex'], json.dumps(r.get('summary_metrics', {}), ensure_ascii=False)])
print(f"CSV guardado en: {csv_path}")
