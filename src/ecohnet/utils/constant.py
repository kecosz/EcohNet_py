PP = (0.95, 0.95, 0.001, 8)
REP = 5000


class Const:
    p1: float = PP[0]
    "Spectral radius of reccurent matrix."

    p2: float = PP[1]
    "Forgetting rate in RLS."

    p3: float = PP[2]
    "Regularize coefficient in RLS."

    p4: int = PP[3]
    "Number of RLS update per one time step."

    q1: int = 32
    "Number of recurrent nodes."

    q2: int = REP
    "Number of ESN population."
