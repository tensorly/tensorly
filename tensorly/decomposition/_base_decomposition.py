"""Base classes for all estimators, ensure compatibility with Scikit-Learn
"""

class DecompositionMixin:
    def fit(self, tensor, **kwargs):
        _ = self.fit_transform(tensor, **kwargs)
        return self
    
    def __repr__(self):
        return f'{self.__class__.__name__} decomposition of rank {self.rank}.'