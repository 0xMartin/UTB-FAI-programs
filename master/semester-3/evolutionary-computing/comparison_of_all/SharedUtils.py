class Range:
    def __init__(self, min: float, max: float) -> None:
        """
        Rozsah hodnot

        Parametry:
            min - minimalni hodnota rozsahu
            max - maximalni hodnota rozsahu
        """
        self.min = min
        self.max = max

    def size(self):
        """
        Vypocita velikost rozsahu
        """
        return self.max - self.min