import numpy as np
from pympler import asizeof

a = {"cls": 1, "bboxes": np.array([0.1, 0.2, 0.3, 0.4])}
b = {"cls": 2, "bboxes": np.array([0.8, 0.6, 0.5, 0.4])}
c = {"cls": 3, "bboxes": np.array([0.4, 0.6, 0.3, 0.1])}
d = [a, b, c]
print(asizeof.asizeof(d))
e = {"cls": np.array([1, 2, 3]), "bboxes": np.array([[0.1, 0.2, 0.3, 0.4], [0.8, 0.6, 0.5, 0.4], [0.4, 0.6, 0.3, 0.1]])}
print(asizeof.asizeof(e))
