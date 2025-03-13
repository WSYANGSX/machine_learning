import torch
from collections import deque
from typing import Iterable


def is_loss_steady(losses: Iterable[float]) -> bool:
    last_loss = losses[-1]
    penult_loss = losses[-2]
    antepenult_loss = losses[-3]

    if (
        abs((antepenult_loss - penult_loss) / antepenult_loss) <= 0.01
        and abs((penult_loss - last_loss) / penult_loss) <= 0.01
    ):
        return True
    else:
        return False


if __name__ == "__main__":
    a = deque(maxlen=5)
    a.append(0.001)
    a.append(0.0012)
    a.append(0.0013)
    b = list(a)
    print(b)
