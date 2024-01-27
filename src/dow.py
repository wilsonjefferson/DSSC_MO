from dataclasses import dataclass
import numpy as np
from gurobipy import tupledict


rng = np.random.default_rng()


@dataclass(frozen=False)
class DOW:

    def __init__(self, m_storages:int, n_fields:int, k_vehicles:int):

        self.m_storages = m_storages
        self.n_fields = n_fields
        self.k_vehicles = k_vehicles

        # NOTE: Since it is very difficult to find a feasible solution
        #   we randmly fix the decision variable X with a random sequence of {0, 1}
        #   then, we use the partial fixed solution to find a feasible solution for the LARP
        #   model. The first feasible solution found is returned as a DOW.

        self._X = None
        self._Y = None
        self._Z = None
        self.obj_value = None

    @property
    def X(self) -> np.array:
        return self._X

    @X.setter
    def X(self, other) -> None:
        self._X = np.array(other)

    @property
    def Y(self) -> np.array:
        return self._Y

    @Y.setter
    def Y(self, other) -> None:
        self._Y = np.zeros((self.n_fields, self.m_storages))
        for i in range(self._Y.shape[0]):
            for j in range(self._Y.shape[1]):
                if isinstance(other, tupledict):
                    self._Y[i,j] = other[i,j].X
                else:
                    self._Y[i,j] = other[i,j]

    @property
    def Z(self) -> np.array:
        return self._Z

    @Z.setter
    def Z(self, other) -> None:
        self._Z = np.zeros((self.m_storages+1, self.m_storages+1))
        for u in range(self._Z.shape[0]):
            for v in range(self._Z.shape[1]):
                if u!=v:
                    if isinstance(other, tupledict):
                        self._Z[u,v] = other[u,v].X
                    else:
                        self._Z[u,v] = other[u,v]
                

    def __str__(self):
        out_string = f'objValue: {self.obj_value}\n'
        out_string += f'X: {self._X}\n'
        out_string += f'Y: {self._Y}\n'
        out_string += f'Z: {self._Z}\n'
        return out_string

    def __eq__(self, other) -> bool:
        if isinstance(other, DOW):
            return self.X == other.X and \
                    self.Y == other.Y \
                    and self.Z == other.Z
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def set_rand_X(self) -> None:
        self._X = rng.integers(2, size=self.m_storages)

    def to_vector(self) -> None:
        self._Y = [np.nonzero(self._Y[i,:])[0][0]+1 for i in range(self._Y.shape[0])]
        self._Y = np.array(self._Y)

        F_firsts_pos = np.nonzero(self._Z[self._Z.shape[0]-1, :] == 1)[0]
        routes = list()
        for pos in F_firsts_pos:
            route = list()
            route.append(0)
            route.append(pos+1)
            while pos != self._Z.shape[1]-1:
                pos = np.nonzero(self._Z[pos, :] == 1)[0][0]
                route.append(pos+1)
            routes.append(route)
        self._Z = [pos for route in routes for pos in route]
        self._Z = np.array(self._Z)
    
    def to_matrix(self) -> None:
        Y = np.zeros((self.n_fields, self.m_storages))
        rows = [i for i in range(len(self._Y))]
        cols = [j-1 for j in self._Y]
        Y[np.ix_(rows, cols)] = 1
        self._Y = Y

        Z = np.zeros((self.m_storages+1, self.m_storages+1))
        for idx in range(len(self._Z)-1):
            u = self._Z[idx]
            v = self._Z[idx+1]
            if u == 0:
                Z[self.m_storages, v] = 1
            else:
                Z[u,v] = 1
        Z[self._Z[len(self._Z)-1],self.m_storages] = 1
        self._Z = Z
        
            