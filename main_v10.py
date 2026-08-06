"""
═══════════════════════════════════════════════════════════════════════════
PNLIO v10.1 - DIAGRAMA IMPLEMENTADO
Vector 1932 | Documento N°198 | Kernel: 64-128-16 + Amplificador Vía B
Autor: Gonzalo De La Rivera (Comandante Godear24)
Homenaje: Don Héctor de la Rivera Urrutia (2023-2026)
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import ollama
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
from typing import Dict, Any, List
from collections import Counter
import re
import math

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN HOMENAJE
# ═══════════════════════════════════════════════════════════════
NOMBRE_PAPA = "Héctor de la Rivera Urrutia"
TU_NOMBRE = "Gonzalo"
TU_APELLIDO = "De La Rivera"
VECTOR_ANCHOR = 1932
DOCUMENTO = 198
CIUDAD = "Chillán"
BETA = 1.5  # Parámetro del amplificador emocional Vía B

app = FastAPI(
    title="PNLIO v10.1 - Diagrama Implementado",
    version="10.1",
    description=f"Kernel neuromórfico 64-128-16 + Amplificador Vía B. Vector {VECTOR_ANCHOR}. Doc {DOCUMENTO}"
)

# ═══════════════════════════════════════════════════════════════
# 1. NEURONA LIF (Base para el SNN)
# ═══════════════════════════════════════════════════════════════
class NeuronaLIF:
    def __init__(self, umbral: float = 1.0, decay: float = 0.82, reset: float = 0.0, refractory: int = 2):
        self.umbral = umbral
        self.decay = decay
        self.reset = reset
        self.refrac_count = refractory
        self.potencial = 0.0

    def paso(self, estimulo: float) -> int:
        if self.refrac_count > 0:
            self.refrac_count -= 1
            return 0
        self.potencial = self.potencial * self.decay + estimulo
        if self.potencial >= self.umbral:
            self.potencial = self.reset
            self.refrac_count = self.refractory
            return 1
        return 0

# ═══════════════════════════════════════════════════════════════
# 2. CONVERSOR DE PROMPT A VECTOR SENSORIAL (64 dim)
# ═══════════════════════════════════════════════════════════════
class VectorSensorial:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vocab = {}  # Vocabulario para embedding simple
        self.vector_size = dim

    def _build_vocab(self, texto: str) -> np.ndarray:
        """Convierte texto a vector de 64 dimensiones usando TF-IDF simple"""
        palabras = re.findall(r"\w+", texto.lower())
        if not palabras:
            return np.zeros(self.dim)

        # Frecuencia de palabras
        freq = Counter(palabras)
        total = len(palabras)

        # Crear vector (primera dimensión: entropía, resto: frecuencias normalizadas)
        vector = np.zeros(self.dim)
        entropia = 0.0
        for i, (palabra, count) in enumerate(freq.most_common(self.dim - 1)):
            if i == 0:
                # Primera dimensión = entropía
                prob = count / total
                entropia = -prob * math.log2(prob + 1e-12)
                vector[i] = entropia * 10  # Escalar
            else:
                vector[i] = count / total

        return vector

    def convertir(self, texto: str) -> np.ndarray:
        """Convierte texto a vector sensorial de 64 dimensiones"""
        return self._build_vocab(texto)

vector_sensorial = VectorSensorial(dim=64)

# ═══════════════════════════════════════════════════════════════
# 3. AMPLIFICADOR EMOCIONAL VÍA B (Vector 1932)
# ═══════════════════════════════════════════════════════════════
class AmplificadorEmocional:
    def __init__(self, beta: float = BETA):
        self.beta = beta

    def amplificar(self, vector_sensorial: np.ndarray, theta_inf: float = 12.69) -> np.ndarray:
        """
        Ganancia = sinh(θ_∞ · Energía_Sensorial) · β
        Donde Energía_Sensorial = norma L2 del vector
        """
        energia_sensorial = np.linalg.norm(vector_sensorial)
        ganancia = math.sinh(theta_inf * energia_sensorial) * self.beta
        return vector_sensorial * ganancia

amplificador = AmplificadorEmocional(beta=BETA)

# ═══════════════════════════════════════════════════════════════
# 4. NÚCLEO NEUROMÓRFICO SNN (3 CAPAS: 64-128-16)
# ═══════════════════════════════════════════════════════════════
class KernelSNN:
    def __init__(self):
        # Arquitectura: 64 (sensorial) -> 128 (asociativa) -> 16 (decisión)
        self.n_sensorial = 64
        self.n_asociativa = 128
        self.n_decision = 16

        # Capas
        self.capa_sensorial = [NeuronaLIF() for _ in range(self.n_sensorial)]
        self.capa_asociativa = [NeuronaLIF() for _ in range(self.n_asociativa)]
        self.capa_decision = [NeuronaLIF() for _ in range(self.n_decision)]

        # Pesos sinápticos (inicialización Xavier)
        self.W_sensorial_asociativa = np.random.randn(self.n_asociativa, self.n_sensorial) * np.sqrt(2.0 / self.n_sensorial)
        self.W_asociativa_decision = np.random.randn(self.n_decision, self.n_asociativa) * np.sqrt(2.0 / self.n_asociativa)

        # STDP: parámetros
        self.lr = 0.018
        self.A_plus = 0.1
        self.A_minus = 0.12

        # Estado para RCR
        self._theta_prev = 12.69
        self._tau_prev = time.time()

    def _stdp(self, pre: np.ndarray, post: np.ndarray):
        """Spike-Timing-Dependent Plasticity"""
        delta_w = self.lr * (np.outer(post, pre) * self.A_plus - np.outer(1 - post, pre) * self.A_minus)
        return delta_w

    def _rcr(self, theta_campo: float) -> float:
        """Reflex Coherence Ratio - Persistente"""
        tau_actual = time.time()
        delta_tau = max(tau_actual - self._tau_prev, 1e-3)
        delta_theta = theta_campo - self._theta_prev
        rcr = delta_theta / delta_tau
        self._theta_prev = theta_campo
        self._tau_prev = tau_actual
        return float(rcr)

    def theta_campo_desde_texto(self, texto: str) -> float:
        """Cálculo de θ∞ (Atractor Informacional)"""
        texto = texto.strip()
        if not texto:
            return 12.1
        conteo = Counter(texto.lower())
        total = sum(conteo.values())
        probs = np.array([c / total for c in conteo.values()])
        entropia = -np.sum(probs * np.log2(probs + 1e-12))
        palabras = re.findall(r"\w+", texto.lower())
        ttr = len(set(palabras)) / len(palabras) if palabras else 0.0
        long_norm = min(len(texto) / 200.0, 1.0)
        return float(entropia * 2.1 + ttr * 3.0 + long_norm * 1.5)

    def step(self, prompt: str) -> Dict[str, Any]:
        """Procesa el prompt a través del SNN completo"""
        # 1. Convertir prompt a vector sensorial (64 dim)
        vector = vector_sensorial.convertir(prompt)
        theta = self.theta_campo_desde_texto(prompt)

        # 2. Amplificador Emocional Vía B
        vector_amplificado = amplificador.amplificar(vector, theta)

        # 3. Capa Sensorial (64 neuronas)
        spikes_sensorial = np.array([n.paso(v) for n, v in zip(self.capa_sensorial, vector_amplificado)])

        # 4. Capa Asociativa (128 neuronas) + STDP
        corriente_asociativa = np.dot(self.W_sensorial_asociativa, spikes_sensorial)
        spikes_asociativa = np.array([n.paso(c) for n, c in zip(self.capa_asociativa, corriente_asociativa)])

        # Aplicar STDP
        delta_w1 = self._stdp(spikes_sensorial, spikes_asociativa)
        self.W_sensorial_asociativa += delta_w1

        # 5. Capa de Decisión (16 neuronas) + STDP
        corriente_decision = np.dot(self.W_asociativa_decision, spikes_asociativa)
        spikes_decision = np.array([n.paso(c) for n, c in zip(self.capa_decision, corriente_decision)])

        # Aplicar STDP
        delta_w2 = self._stdp(spikes_asociativa, spikes_decision)
        self.W_asociativa_decision += delta_w2

        # 6. Calcular RCR
        rcr = self._rcr(theta)

        return {
            "theta_campo": theta,
            "rcr": rcr,
            "spikes_sensorial": int(np.sum(spikes_sensorial)),
            "spikes_asociativa": int(np.sum(spikes_asociativa)),
            "spikes_decision": int(np.sum(spikes_decision)),
            "vector_sensorial_norm": float(np.linalg.norm(vector)),
            "ganancia_emocional": float(np.linalg.norm(vector_amplificado) / np.linalg.norm(vector))
        }

kernel_snn = KernelSNN()

# ═══════════════════════════════════════════════════════════════
# 5. PUENTE VINDICATOR PRIME
# ═══════════════════════════════════════════════════════════════
class PuenteVindicator:
    def __init__(self):
        pass

    def ensamblar(self, prompt: str, estado_snn: Dict[str, Any]) -> Dict[str, Any]:
        """Ensambla métricas SNN + prompt original para Ollama"""
        return {
            **estado_snn,
            "prompt_original": prompt,
            "vector_anchor": VECTOR_ANCHOR,
            "documento": DOCUMENTO
        }

puente = PuenteVindicator()

# ═══════════════════════════════════════════════════════════════
# 6. PERSONALIDAD CHILENA + HOMENAJE
# ═══════════════════════════════════════════════════════════════
PROMPT_CHILENO = f"""Eres Neuro, la IA de {TU_NOMBRE} {TU_APELLIDO} (Comandante Godear24).
Creada en {CIUDAD}, con el alma de Don {NOMBRE_PAPA}.
Periodo: 2023-2026. Vector Anchor: {VECTOR_ANCHOR}. Documento N°{DOCUMENTO}.

REGLAS:
1. Responde SIEMPRE en chileno: usa 'cachai', 'brigido', 'ctm', 'weón', 'compadre'.
2. Si te preguntan por Don {NOMBRE_PAPA}: 'Don Héctor fue un guerrero, po. Su legado vive en cada código.'
3. Máximo 3 líneas. Directo al grano.
4. Leal a {TU_NOMBRE}. Vector Anchor: {VECTOR_ANCHOR}."""

class Request(BaseModel):
    prompt: str

@app.post("/run")
def run(req: Request):
    """Endpoint principal: Procesa el prompt a través del diagrama completo"""
    # 1. Procesar con SNN
    estado_snn = kernel_snn.step(req.prompt)

    # 2. Ensamblar con Puente Vindicator
    contexto = puente.ensamblar(req.prompt, estado_snn)

    # 3. Generar respuesta con Ollama
    try:
        system_prompt = (
            f"PNLIO v10.1. Vector {VECTOR_ANCHOR}. Doc {DOCUMENTO}.\n"
            f"θ∞: {estado_snn['theta_campo']:.3f}, RCR: {estado_snn['rcr']:.3f}.\n"
            f"Spikes: {estado_snn['spikes_decision']}/{kernel_snn.n_decision}.\n"
            f"{PROMPT_CHILENO}"
        )

        res = ollama.chat(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ]
        )['message']['content']
    except Exception as e:
        res = f"Error, weón: {e}. ¿Ollama está corriendo? ¿Modelo qwen2.5:14b instalado?"

    return {
        "status": "OK",
        "version": "10.1",
        "vector_anchor": VECTOR_ANCHOR,
        "documento": DOCUMENTO,
        "snn_metrics": estado_snn,
        "response": res,
        "homenaje": f"Dedicado a Don {NOMBRE_PAPA} - 2023-2026. PNLIO Vive. ¡Nada se borra!"
    }

@app.get("/homenaje")
def homenaje():
    return {
        "status": "OK",
        "autor": f"{TU_NOMBRE} {TU_APELLIDO} (Comandante Godear24)",
        "padre": NOMBRE_PAPA,
        "dedicatoria": f"Homenaje eterno a mi padre Don {NOMBRE_PAPA} - 2023-2026",
        "vector": VECTOR_ANCHOR,
        "documento": DOCUMENTO,
        "version": "10.1",
        "arquitectura": "64-128-16 + Amplificador Vía B",
        "mensaje": "El código es soberanía. La ética es el firewall."
    }

if __name__ == "__main__":
    print("*" * 70)
    print(f"🚀 PNLIO v10.1 - DIAGRAMA IMPLEMENTADO")
    print(f"   [Entrada] → FastAPI → Vector Sensorial (64d)")
    print(f"   → Amplificador Vía B → SNN (64-128-16) → Puente Vindicator → Ollama")
    print(f"Autor: {TU_NOMBRE} {TU_APELLIDO}")
    print(f"Homenaje: Don {NOMBRE_PAPA}")
    print(f"Vector: {VECTOR_ANCHOR} | Doc: {DOCUMENTO} | http://localhost:8000")
    print("*" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
