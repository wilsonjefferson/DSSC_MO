from dataclasses import dataclass
import numpy as np


rng = np.random.default_rng()


@dataclass(frozen=False)
class DOW:

    def __init__(self, m_storages:int, n_fields:int, k_vehicles:int):

        self.m_storages = m_storages
        self.n_fields = n_fields
        self.k_vehicles = k_vehicles

        self.l_storages, index_storages = self._rand_storages()

        # NOTE: Since it is very difficult to find a feasible solution
        #   we randmly fix the decision variable X with a random sequence of {0, 1}
        #   then, we use the partial fixed solution to find a feasible solution for the LARP
        #   model. The first feasible solution found is returned as a DOW.

        self.X = self.l_storages
        self.Y = None
        self.Z = None

        self.obj_value = None

    def __str__(self):
        out_string = f'objValue: {self.obj_value}\n\n'
        out_string += f'X: {self.X}\n\n'
        out_string += f'Y: {self.Y}\n\n'
        out_string += f'Z: {self.Z}\n\n'
        return out_string

    def __eq__(self, other):
        if isinstance(other, DOW):
            return self.X == other.X and \
                    self.Y == other.Y \
                    and self.Z == other.Z
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _rand_storages(self):
        l_storages = rng.integers(2, size=self.m_storages)
        index_storages = [i+1 for i in np.nonzero(l_storages)[0]]
        return l_storages, index_storages
            