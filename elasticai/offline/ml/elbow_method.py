# -----------------------------------------------------------------------------------------------------------------------
# Input: frames.waveform -> als array?
# Output: summary.csv, plots, features-cache
# Config: Input, welche pipeline stufe, welche fex methoden, welche cluster methoden, welche k-werte, welche metriken?
# output caching einstellungen
# später (eventuell): artefakt handling, preprocessing varianten, normalisierung
# -----------------------------------------------------------------------------------------------------------------------


class ElbowRunner:
    def __init__(
        self,
    ):
        self.fex_methods = list()
        self.cluster_methods = list()

    def compute_embeddings(self):
        pass

    def load_embedding(self):
        pass

    def run_clustering(self):
        pass

    def evaluate(self):
        pass

    def save_summary(self):
        pass

    def plot_results(self):
        pass

    def run(self):
        pass
