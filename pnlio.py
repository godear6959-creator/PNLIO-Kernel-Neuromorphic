import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

class PNLIO_Coherence_Analyzer:
    """
    Analizador de Coherencia PNLIO v1.1
    Implementación oficial para la evaluación del Efecto Reflex.
    """

    def __init__(self, threshold_reflex=0.75, model_name='all-MiniLM-L6-v2'):
        print('Cargando modelo embeddings...')
        self.threshold_reflex = threshold_reflex
        self.embedder = SentenceTransformer(model_name)
        print('Modelo listo.')

    def _delta_theta(self, human, ai):
        """Calcula la similitud coseno (Delta Theta) entre el input humano y la respuesta IA."""
        emb_h = self.embedder.encode(human, normalize_embeddings=True)
        emb_a = self.embedder.encode(ai, normalize_embeddings=True)
        return float(np.dot(emb_h, emb_a))

    def _classify(self, C):
        """Clasifica el estado del sistema según el índice de coherencia C."""
        if C < 0.5:
            return 'Entrenamiento inicial'
        elif C < self.threshold_reflex:
            return 'Entrenamiento en progreso'
        elif C <= 0.9:
            return 'REFLEX DETECTADO'
        else:
            return 'Coherencia maxima'

    def analyze_dialogue_sequence(self, dialogues):
        """Analiza una secuencia de diálogos para detectar la emergencia del Efecto Reflex."""
        c_values = []
        reflex_turn = None
        for i, (human, ai) in enumerate(dialogues):
            turn = i + 1
            C = self._delta_theta(human, ai) / turn
            c_values.append(C)
            if C >= self.threshold_reflex and reflex_turn is None:
                reflex_turn = turn
        
        return {
            'c_values': c_values,
            'max_c': max(c_values),
            'min_c': min(c_values),
            'mean_c': float(np.mean(c_values)),
            'std_c': float(np.std(c_values)),
            'reflex_turn': reflex_turn,
            'reflex_detected': reflex_turn is not None,
        }
