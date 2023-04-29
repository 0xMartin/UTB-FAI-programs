from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class CMResult():
    """
    Datova trida s vysledky z klasifikace
    """

    # celkovy pocet testovanych zaznamu v datasetu
    total_cnt: int

    # pocet chybne predikovanych zaznamu v datasetu
    incorrect_cnt: int

    # score predikovani (% uspesnost)
    score: float


class ClassificationModel(ABC):
    """
    Rozhrani pro tridy klasifikatoru
    """

    @abstractmethod
    def trainModel(self, df_train: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def classify(self, df_test: pd.DataFrame) -> CMResult:
        pass
