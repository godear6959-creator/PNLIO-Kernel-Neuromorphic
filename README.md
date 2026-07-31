![Imagen Principal](gonzalo-mauricio-de-la-rivera-arellano-geminis-imagen-generada-4zd3tz4zd3tz4zd3.png)
# Núcleo de inferencia neuromórfica (NIK) v10.1 — Edición universal

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

**NIK v10.1** es un núcleo de inferencia neuromórfica local diseñado para operar con arquitectura *offline-first*. Integra una red neuronal de impulsos (Spiking Neural Network - SNN) de tres capas (64-128-16) con aprendizaje biológico adaptativo, regulación homeostática y un puente de integración directa con modelos de lenguaje locales ejecutados en Ollama.

---

## 🏛️ Información del Autor y Proyecto

- **Autor:** Gonzalo Mauricio de la Rivera Arellano
- **GitHub:** godear6959-creator
- **ORCID:** `0009-0001-9455-8416`
- **Ubicación:** Chillán, Ñuble, Chile
- **Licencia:** MIT

> **DEDICATORIA:** Homenaje eterno a mi padre (2023-2026). Este proyecto es un acto de soberanía tecnológica y memoria.

---

## ⚙️ Especificaciones de la Arquitectura

1. **Red Neuronal de Impulsos (SNN 64-128-16):**
   - **Capa Sensorial (L1):** 64 neuronas LIF (Leaky Integrate-and-Fire).
   - **Capa Asociativa (L2):** 128 neuronas LIF.
   - **Capa de Decisión (L3):** 16 neuronas LIF.
2. **Plasticidad STDP (Spike-Timing-Dependent Plasticity):**
   - Implementación de ventana exponencial temporal real basada en los tiempos del último disparo (`last_spike_t`).
3. **Regulación Homeostática:**
   - Ajuste dinámico de umbrales neuronales según la tasa de disparo para prevenir saturación o inactividad.
4. **Vía de Modulación Emocional (EMP / Vía B):**
   - Ganancia dinámica de señal basada en entropía y tasa léxica para la amplificación de estímulos.
5. **Puente LLM Local (Ollama):**
   - Inyección del estado dinámico de la SNN dentro del contexto del sistema de Ollama (`qwen2.5:14b` por defecto).
6. **Módulo de Búsqueda Web Local:**
   - Consulta web integrada de respaldo (DuckDuckGo HTML) para proporcionar contexto actualizado cuando sea requerido.

---

## 🚀 Instalación y Requisitos

### Requisitos Previos

- Python 3.10 o superior
- Ollama instalado y ejecutándose localmente

```bash
# Descargar e instalar el modelo por defecto en Ollama
ollama pull qwen2.5:14b
