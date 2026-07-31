"""
═══════════════════════════════════════════════════════════════════════
NEUROMORPHIC INFERENCE KERNEL (NIK) v10.1 — Universal Edition
Architecture: 64-128-16 SNN + Emotional Modulation Pathway + Local LLM Bridge
License: MIT | Scientific Personality | Offline-First Design
═══════════════════════════════════════════════════════════════════════
CHANGES v10.0 -> v10.1:
  [FIX] _stdp() now genuinely uses an exponential time window (STDP_TAU),
        based on each neuron's real last_spike_t — previously the docstring
        promised an "exponential window" but the function only computed an
        instantaneous same-timestep outer product (plain Hebbian, no timing).
  [FIX] use_web_search=True is now respected even when the prompt doesn't
        match the internal keyword heuristic. Previously an explicit True
        flag was silently overridden by needs_search() inside search().
  [CAL] SENSORY_GAIN=3.0 and INTER_LAYER_GAIN=3.5 added empirically so
        spikes actually reach L2/L3 instead of dying out at the sensory layer.
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import ollama
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
from typing import Dict, Any, List, Optional
from collections import Counter
from html import unescape
import re
import math
from urllib.parse import quote

# ═══════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
APP_NAME = "Neuromorphic Inference Kernel (NIK) v10.1"
VERSION = "10.1.0-universal"
DEFAULT_MODEL = "qwen2.5:14b"

# SNN Hyperparameters
SENSORIAL_DIM = 64
ASSOCIATIVE_DIM = 128
DECISION_DIM = 16
LIF_DECAY = 0.82
LIF_THRESHOLD = 1.0
LIF_REFRACTORY = 2
CLIP_VALUE = 2.0

# STDP Hyperparameters
STDP_LR = 0.018
STDP_A_PLUS = 0.08
STDP_A_MINUS = 0.04
STDP_TAU = 20.0

# Homeostasis
HOMEO_TARGET_RATE = 0.015  # spikes per ms
HOMEO_RATE = 0.002

# Emotional Modulation Pathway (EMP)
EMP_BETA = 1.5
EMP_THETA_DEFAULT = 12.69

# Signal propagation gains [NEW v10.1] — calibrated empirically so spikes
# actually reach L2/L3 instead of dying out at the sensory layer.
SENSORY_GAIN = 3.0       # amplifies encoded input before hitting L1
INTER_LAYER_GAIN = 3.5   # amplifies L1->L2 and L2->L3 synaptic currents

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="Local neuromorphic kernel 64-128-16 with real-window STDP, homeostasis, and LLM bridge."
)


# ═══════════════════════════════════════════════════════════════════
# 1. LIF NEURON (Leaky Integrate-and-Fire)
# ═══════════════════════════════════════════════════════════════════
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron with refractory period.

    V(t+1) = V(t) * decay + I(t)
    spike if V >= threshold, then V = reset for refractory steps
    """
    def __init__(self, threshold: float = LIF_THRESHOLD, decay: float = LIF_DECAY,
                 reset: float = 0.0, refractory: int = LIF_REFRACTORY):
        self.threshold = threshold
        self.decay = decay
        self.reset = reset
        self.refractory_period = refractory
        self.refrac_counter = 0
        self.potential = 0.0
        self.spike_count = 0
        self.last_spike_t = -9999.0

    def step(self, stimulus: float, t: float) -> int:
        """Advance one time step. Returns 1 if spike occurred, 0 otherwise."""
        if self.refrac_counter > 0:
            self.refrac_counter -= 1
            return 0

        self.potential = self.potential * self.decay + stimulus

        if self.potential >= self.threshold:
            self.potential = self.reset
            self.refrac_counter = self.refractory_period
            self.spike_count += 1
            self.last_spike_t = t
            return 1
        return 0


# ═══════════════════════════════════════════════════════════════════
# 2. SENSORY VECTOR ENCODER (64-DIM)
# ═══════════════════════════════════════════════════════════════════
class SensoryEncoder:
    """
    Converts text into a 64-dimensional sensory vector.

    Dimension 0: Shannon entropy of word frequency (×10)
    Dimensions 1-63: Normalized frequency of most common words
    """
    def __init__(self, dim: int = SENSORIAL_DIM):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        words = re.findall(r"\w+", text.lower())
        if not words:
            return np.zeros(self.dim)

        freq = Counter(words)
        total = len(words)
        vector = np.zeros(self.dim)

        # Entropy feature (dimension 0)
        top_count = freq.most_common(1)[0][1]
        p_top = top_count / total
        entropy = -p_top * math.log2(p_top + 1e-12)
        vector[0] = entropy * 10.0

        # Word frequency features (dimensions 1-63)
        for i, (_, count) in enumerate(freq.most_common(self.dim - 1)):
            vector[i + 1] = count / total

        return vector


# ═══════════════════════════════════════════════════════════════════
# 3. EMOTIONAL MODULATION PATHWAY (EMP / "Vía B")
# ═══════════════════════════════════════════════════════════════════
class EmotionalModulator:
    """
    Modulates sensory input based on field coherence (theta).

    Uses tanh-scaled amplification to prevent numerical overflow
    while preserving non-linear emotional gain.
    """
    def __init__(self, beta: float = EMP_BETA):
        self.beta = beta

    def modulate(self, vector: np.ndarray, theta: float) -> np.ndarray:
        energy = np.linalg.norm(vector)
        # tanh prevents overflow; preserves sigmoidal emotional response
        gain = math.tanh(theta * energy * 0.1) * self.beta
        gain = max(gain, 0.1)  # minimum 0.1x amplification
        return vector * gain * SENSORY_GAIN


# ═══════════════════════════════════════════════════════════════════
# 4. SNN KERNEL (64 → 128 → 16) + STDP + HOMEOSTASIS
# ═══════════════════════════════════════════════════════════════════
class SNNKernel:
    """
    Three-layer spiking neural network with Hebbian STDP and
    activity-dependent homeostatic threshold regulation.

    Layers:
        L1: Sensory   (64 neurons)  — receives encoded text
        L2: Associative (128 neurons) — feature extraction
        L3: Decision  (16 neurons)  — output / routing signals
    """
    def __init__(self):
        self.n_s = SENSORIAL_DIM
        self.n_a = ASSOCIATIVE_DIM
        self.n_d = DECISION_DIM

        # Neuron populations
        self.layer_s = [LIFNeuron() for _ in range(self.n_s)]
        self.layer_a = [LIFNeuron() for _ in range(self.n_a)]
        self.layer_d = [LIFNeuron() for _ in range(self.n_d)]

        # Synaptic weights (He initialization)
        self.W_sa = np.random.randn(self.n_a, self.n_s) * np.sqrt(2.0 / self.n_s)
        self.W_ad = np.random.randn(self.n_d, self.n_a) * np.sqrt(2.0 / self.n_a)

        # STDP parameters
        self.lr = STDP_LR
        self.A_plus = STDP_A_PLUS
        self.A_minus = STDP_A_MINUS
        self.tau_stdp = STDP_TAU

        # Homeostasis
        self.homeo_target = HOMEO_TARGET_RATE
        self.homeo_rate = HOMEO_RATE

        # RCR tracking
        self._theta_prev = EMP_THETA_DEFAULT
        self._tau_prev = time.monotonic()
        self.t_global = 0.0

        # Encoder & modulator
        self.encoder = SensoryEncoder()
        self.modulator = EmotionalModulator()

    # ── STDP rule with real exponential window ──
    def _stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
              pre_last_spike_before: np.ndarray, post_last_spike_before: np.ndarray) -> np.ndarray:
        """
        [FIX v10.1] Real exponential-window STDP using last_spike_t.
        Potentiation: post fires now, weighted by recency of each pre's last spike.
        Depression: pre fires now, weighted by recency of each post's last spike.
        Neurons that never fired (sentinel -9999.0) contribute zero.
        """
        t = self.t_global
        never_fired = -9000.0

        dt_pot = t - pre_last_spike_before
        pot_window = np.where(pre_last_spike_before > never_fired,
                               np.exp(-dt_pot / self.tau_stdp), 0.0)
        potentiation = np.outer(post_spikes, pot_window) * self.A_plus

        dt_dep = t - post_last_spike_before
        dep_window = np.where(post_last_spike_before > never_fired,
                               np.exp(-dt_dep / self.tau_stdp), 0.0)
        depression = np.outer(dep_window, pre_spikes) * self.A_minus

        return self.lr * (potentiation - depression)

    # ── RCR metric ──
    def _compute_rcr(self, theta: float) -> float:
        """
        Rate of Coherence Change (RCR).
        Measures how rapidly the semantic field (theta) evolves.
        Uses monotonic clock for stable delta-time measurement.
        """
        tau_now = time.monotonic()
        delta_tau = max(tau_now - self._tau_prev, 1e-3)
        delta_theta = theta - self._theta_prev
        rcr = delta_theta / delta_tau
        self._theta_prev = theta
        self._tau_prev = tau_now
        return float(rcr)

    # ── Semantic field coherence (theta) ──
    def _compute_theta(self, text: str) -> float:
        """
        Computes semantic field coherence from text statistics.
        Combines character entropy, type-token ratio, and length normalization.
        """
        text = text.strip()
        if not text:
            return EMP_THETA_DEFAULT

        # Character-level entropy
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        probs = np.array([c / total_chars for c in char_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # Type-token ratio (lexical diversity)
        words = re.findall(r"\w+", text.lower())
        ttr = len(set(words)) / len(words) if words else 0.0

        # Length normalization (saturates at 200 chars)
        length_norm = min(len(text) / 200.0, 1.0)

        return float(entropy * 2.1 + ttr * 3.0 + length_norm * 1.5)

    # ── Homeostatic threshold adjustment ──
    def _homeostasis(self):
        """
        Adjusts firing thresholds based on recent activity.
        High activity → higher threshold (slows down)
        Low activity → lower threshold (sensitizes)
        """
        time_ms = self.t_global + 1.0

        for layer in [self.layer_s, self.layer_a, self.layer_d]:
            for neuron in layer:
                rate = neuron.spike_count / time_ms
                neuron.threshold += self.homeo_rate * (rate - self.homeo_target) * 10.0
                neuron.threshold = max(0.4, min(2.5, neuron.threshold))

    # ── Weight normalization ──
    def _normalize_weights(self):
        """Hard clip to prevent numerical divergence."""
        self.W_sa = np.clip(self.W_sa, -CLIP_VALUE, CLIP_VALUE)
        self.W_ad = np.clip(self.W_ad, -CLIP_VALUE, CLIP_VALUE)

    # ── Main forward pass ──
    def step(self, prompt: str) -> Dict[str, Any]:
        """
        Single forward pass through the neuromorphic pipeline.

        Returns dict with theta, RCR, spike counts, and gain metrics.
        """
        self.t_global += 1.0

        # 1. Encode text to sensory vector
        vector = self.encoder.encode(prompt)

        # 2. Compute semantic field coherence
        theta = self._compute_theta(prompt)

        # 3. Emotional modulation (Vía B)
        vector_mod = self.modulator.modulate(vector, theta)

        # 4. Capture PRE-step last spike times for real STDP window
        prev_last_s = np.array([n.last_spike_t for n in self.layer_s])
        prev_last_a = np.array([n.last_spike_t for n in self.layer_a])
        prev_last_d = np.array([n.last_spike_t for n in self.layer_d])

        # 5. Layer 1: Sensory (spike encoding)
        spikes_s = np.array([n.step(v, self.t_global) for n, v in zip(self.layer_s, vector_mod)])

        # 6. Layer 2: Associative (feedforward + STDP)
        current_a = np.dot(self.W_sa, spikes_s) * INTER_LAYER_GAIN
        spikes_a = np.array([n.step(c, self.t_global) for n, c in zip(self.layer_a, current_a)])
        self.W_sa += self._stdp(spikes_s, spikes_a, prev_last_s, prev_last_a)

        # 7. Layer 3: Decision (feedforward + STDP)
        current_d = np.dot(self.W_ad, spikes_a) * INTER_LAYER_GAIN
        spikes_d = np.array([n.step(c, self.t_global) for n, c in zip(self.layer_d, current_d)])
        self.W_ad += self._stdp(spikes_a, spikes_d, prev_last_a, prev_last_d)

        # 8. Homeostasis & normalization
        self._homeostasis()
        self._normalize_weights()

        # 9. Compute RCR
        rcr = self._compute_rcr(theta)

        # 10. Metrics
        norm_orig = np.linalg.norm(vector)
        norm_mod = np.linalg.norm(vector_mod)
        gain = float(norm_mod / norm_orig) if norm_orig > 0 else 1.0

        return {
            "theta_field": round(theta, 4),
            "rcr": round(rcr, 4),
            "spikes_sensorial": int(np.sum(spikes_s)),
            "spikes_associative": int(np.sum(spikes_a)),
            "spikes_decision": int(np.sum(spikes_d)),
            "vector_norm": round(float(norm_orig), 4),
            "emotional_gain": round(gain, 4),
            "mean_threshold_l1": round(np.mean([n.threshold for n in self.layer_s]), 4),
            "mean_threshold_l2": round(np.mean([n.threshold for n in self.layer_a]), 4),
            "mean_threshold_l3": round(np.mean([n.threshold for n in self.layer_d]), 4),
        }


# ═══════════════════════════════════════════════════════════════════
# 5. WEB SEARCH MODULE (DuckDuckGo)
# ═══════════════════════════════════════════════════════════════════
class WebSearchModule:
    """
    Lightweight web search via DuckDuckGo HTML interface.
    Automatically detects queries that benefit from external data.
    """
    KEYWORDS = [
        "what is", "who is", "how to", "when", "where", "why",
        "news", "current", "price of", "value of", "latest",
        "weather", "time", "schedule", "date", "today", "now",
        "statistics", "data", "information about"
    ]

    @staticmethod
    def needs_search(prompt: str) -> bool:
        """Heuristic: does this prompt likely require up-to-date external data?"""
        p = prompt.lower().strip()
        if p.endswith("?"):
            return True
        return any(kw in p for kw in WebSearchModule.KEYWORDS)

    @staticmethod
    def search(query: str, max_results: int = 3, force: bool = False) -> str:
        """
        [FIX v10.1] Fetch search snippets from DuckDuckGo.
        `force=True` bypasses needs_search() — makes an explicit
        `use_web_search: true` in the API actually take effect.
        """
        if not force and not WebSearchModule.needs_search(query):
            return ""

        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NIK-v10"}
        try:
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                matches = re.findall(
                    r'<a class="result__snippet[^>]*>(.*?)</a>', res.text, re.DOTALL
                )
                snippets = [
                    unescape(re.sub(r'<[^>]+>', '', m)).strip()
                    for m in matches[:max_results]
                ]
                return "\n".join(f"- {s}" for s in snippets) if snippets else ""
        except Exception as e:
            return f"[Search error: {str(e)}]"
        return ""


# ═══════════════════════════════════════════════════════════════════
# 6. LLM BRIDGE
# ═══════════════════════════════════════════════════════════════════
class LLMBridge:
    """
    Bridges the SNN state to a local LLM (Ollama).
    Injects neuromorphic metrics into the system prompt for
    context-aware, scientifically grounded responses.
    """
    SYSTEM_PERSONA = """You are an AI research assistant powered by a neuromorphic inference kernel.
You analyze queries using a 64-128-16 spiking neural network with real-window STDP plasticity and homeostatic regulation.

Rules:
1. Respond with scientific precision. Cite concepts clearly.
2. When web context is provided, integrate it rigorously.
3. Be concise but thorough. Avoid speculation beyond the evidence.
4. Acknowledge uncertainty when data is insufficient.
5. Maintain a neutral, analytical tone."""

    def query(self, prompt: str, snn_state: Dict[str, Any],
              web_context: str = "", model: str = DEFAULT_MODEL) -> str:
        """Send prompt to local LLM with SNN state as system context."""
        system_msg = (
            f"{self.SYSTEM_PERSONA}\n\n"
            f"[Neuromorphic State]\n"
            f"  Semantic field (θ): {snn_state['theta_field']}\n"
            f"  Coherence rate (RCR): {snn_state['rcr']}\n"
            f"  Spiking activity: L1={snn_state['spikes_sensorial']}, "
            f"L2={snn_state['spikes_associative']}, L3={snn_state['spikes_decision']}\n"
            f"  Emotional gain: {snn_state['emotional_gain']}x\n"
        )
        if web_context:
            system_msg += f"\n[Web Context]\n{web_context}\n"

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return (
                f"LLM bridge error: {e}.\n"
                f"Ensure Ollama is running and model '{model}' is installed."
            )


# ═══════════════════════════════════════════════════════════════════
# 7. FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════
kernel = SNNKernel()
searcher = WebSearchModule()
bridge = LLMBridge()


class InferenceRequest(BaseModel):
    prompt: str
    use_web_search: Optional[bool] = None
    model: Optional[str] = DEFAULT_MODEL


@app.post("/inference")
def inference(req: InferenceRequest):
    """
    Main inference endpoint.

    Pipeline: prompt → SNN kernel → [optional web search] → LLM bridge → response

    [FIX v10.1] use_web_search semantics, now unambiguous:
        None  -> auto-detect via keyword heuristic
        True  -> ALWAYS search, regardless of heuristic (force=True)
        False -> NEVER search
    """
    # 1. Neuromorphic processing
    snn_state = kernel.step(req.prompt)

    # 2. Optional web search
    if req.use_web_search is None:
        web_ctx = searcher.search(req.prompt)
    elif req.use_web_search:
        web_ctx = searcher.search(req.prompt, force=True)
    else:
        web_ctx = ""

    # 3. LLM generation
    llm_response = bridge.query(
        prompt=req.prompt,
        snn_state=snn_state,
        web_context=web_ctx,
        model=req.model or DEFAULT_MODEL
    )

    return {
        "status": "success",
        "version": VERSION,
        "snn_state": snn_state,
        "web_context": web_ctx,
        "response": llm_response
    }


@app.get("/health")
def health():
    """System health and kernel metadata."""
    return {
        "status": "healthy",
        "kernel": APP_NAME,
        "version": VERSION,
        "architecture": f"{SENSORIAL_DIM}-{ASSOCIATIVE_DIM}-{DECISION_DIM} SNN",
        "features": ["LIF neurons", "STDP plasticity (real exp. window)", "Homeostasis",
                      "EMP modulation", "Web search", "Ollama bridge"],
        "default_model": DEFAULT_MODEL,
        "message": "Local neuromorphic inference kernel operational."
    }


@app.get("/snn/state")
def snn_state():
    """Returns current SNN internal state for inspection."""
    return {
        "weights_sa_shape": kernel.W_sa.shape,
        "weights_sa_mean": round(float(kernel.W_sa.mean()), 6),
        "weights_sa_std": round(float(kernel.W_sa.std()), 6),
        "weights_ad_shape": kernel.W_ad.shape,
        "weights_ad_mean": round(float(kernel.W_ad.mean()), 6),
        "weights_ad_std": round(float(kernel.W_ad.std()), 6),
        "thresholds_l1_mean": round(np.mean([n.threshold for n in kernel.layer_s]), 4),
        "thresholds_l2_mean": round(np.mean([n.threshold for n in kernel.layer_a]), 4),
        "thresholds_l3_mean": round(np.mean([n.threshold for n in kernel.layer_d]), 4),
        "total_spikes_l1": sum(n.spike_count for n in kernel.layer_s),
        "total_spikes_l2": sum(n.spike_count for n in kernel.layer_a),
        "total_spikes_l3": sum(n.spike_count for n in kernel.layer_d),
        "global_time": kernel.t_global,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print(f"  {APP_NAME}")
    print(f"  Architecture: {SENSORIAL_DIM} → {ASSOCIATIVE_DIM} → {DECISION_DIM} SNN")
    print(f"  Features: LIF | STDP (real exp. window) | Homeostasis | EMP | Web Search | Ollama Bridge")
    print(f"  Default LLM: {DEFAULT_MODEL}")
    print(f"  Endpoint: http://localhost:8000/inference")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
