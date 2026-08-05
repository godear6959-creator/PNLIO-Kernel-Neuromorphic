PNLIO — Ejemplos para kernel-neuromorfico

Este repositorio contiene ejemplos sencillos para usar el analizador PNLIO incluida una versión que guarda la gráfica automáticamente y un script para ejecutar evaluaciones en lote.

Archivos añadidos/modificados:

- examples/pnlio_analyzer_example.py: ejemplo que ejecuta analyzer.print_report(...) y guarda los resultados en examples/output/ (JSON + PNG si la gráfica está disponible).
- examples/pnlio_batch_evaluate.py: recorre varios modelos y umbrales, guarda batch_results.json y batch_results.csv en examples/output/.
- requirements.txt: dependencias mínima para ejecutar los ejemplos.

Cómo ejecutar

1) Clona el repositorio y entra en la carpeta:
   git clone https://github.com/godear6959-creator/kernel-neuromorfico
   cd kernel-neuromorfico

2) Crear y activar un entorno virtual:
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate

3) Instalar dependencias:
   pip install -r requirements.txt
   # Si PNLIO es el paquete de este repositorio, puede que prefieras instalar en editable:
   # pip install -e .

4) Ejecutar el ejemplo simple (guarda PNG y JSON si es posible):
   python examples/pnlio_analyzer_example.py

5) Ejecutar el batch (genera batch_results.json y batch_results.csv):
   python examples/pnlio_batch_evaluate.py

Notas

- Los scripts intentan ser tolerantes a distintas implementaciones de print_report: si el resultado contiene un objeto 'figure' lo guardan; si PNLIO usa matplotlib internamente, también intentan capturar la figura actual.
- Ajusta la lista de modelos y umbrales en examples/pnlio_batch_evaluate.py según tus necesidades.
