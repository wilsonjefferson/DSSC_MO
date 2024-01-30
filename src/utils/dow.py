from dataclasses import dataclass
from copy import deepcopy
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
        self._obj_value = None

    @property
    def obj_value(self):
        return self._obj_value

    @obj_value.setter
    def obj_value(self, other):
        self._obj_value = deepcopy(other)

    @property
    def X(self) -> np.array:
        return self._X

    @X.setter
    def X(self, other) -> None:
        self._X = np.array(deepcopy(other))

    @property
    def Y(self) -> np.array:
        return self._Y

    @Y.setter
    def Y(self, other) -> None:
        if isinstance(other, np.ndarray):
            self._Y = other
        else:
            self._Y = np.zeros((self.n_fields, self.m_storages))
            for i in range(self._Y.shape[0]):
                for j in range(self._Y.shape[1]):
                    if isinstance(other, tupledict):
                        self._Y[i,j] = deepcopy(other[i,j].x)
                    else:
                        self._Y[i,j] = deepcopy(other[i,j])

    @property
    def Z(self) -> np.array:
        return self._Z

    @Z.setter
    def Z(self, other) -> None:
        if isinstance(other, np.ndarray):
            self._Z = other
        else:
            self._Z = np.zeros((self.m_storages+1, self.m_storages+1))
            for u in range(self._Z.shape[0]):
                for v in range(self._Z.shape[1]):
                    if u!=v:
                        if isinstance(other, tupledict):
                            self._Z[u,v] = deepcopy(other[u,v].x)
                        else:
                            self._Z[u,v] = deepcopy(other[u,v])
                

    def __str__(self):
        out_string = f'objValue: {self._obj_value}\n'
        out_string += f'X: {self._X}\n'
        out_string += f'Y: {self._Y}\n'
        out_string += f'Z: {self._Z}\n'
        return out_string
    
    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other) -> bool:        
        if isinstance(other, DOW):
            if all(arr is not None for arr in [other.Z, other.Y, self._Z, self._Y]):
                other.to_vector()
                self.to_vector()

                return (self.X == other.X).all() and \
                        (self.Y == other.Y).all() and \
                        (len(self.Z) == len(other.Z)) and \
                        (self.Z == other.Z).all()

        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def set_rand_X(self) -> None:
        self._X = rng.integers(2, size=self.m_storages)

    def to_vector(self) -> None:
        if self._Y.ndim == 2:
            self._Y = [np.nonzero(self._Y[i,:])[0][0]+1 for i in range(self._Y.shape[0])]
            self._Y = np.array(self._Y)
        
        if self._Z.ndim == 2:
            F_firsts_pos = np.nonzero(self._Z[self._Z.shape[0]-1, :] == 1)[0]
            # print('F firsts pos:', F_firsts_pos)
            routes = list()
            for pos in F_firsts_pos:
                route = list()
                route.append(0)
                route.append(pos+1)
                while True:
                    pos = np.nonzero(self._Z[pos, :] == 1)[0][0]
                    # print('while | pos:', pos)
                    if pos == self._Z.shape[1]-1:
                        break
                    route.append(pos+1)
                routes.append(route)
            self._Z = [pos for route in routes for pos in route]
            self._Z = np.array(self._Z)
    
    def to_matrix(self) -> None:
        if self._Y.ndim == 1:
            Y = np.zeros((self.n_fields, self.m_storages))
            cols = [j-1 for j in self._Y]
            for i in range(len(self._Y)):
                
                j = cols[i]
                try:
                    Y[i,j] = 1
                except IndexError as e:
                    print(self._Y)
                    print('i:', i, 'j:', j)
                    raise(e)
            self._Y = Y

        if self._Z.ndim == 1:
            Z = np.zeros((self.m_storages+1, self.m_storages+1))
            for idx in range(len(self._Z)-1):
                u = self._Z[idx]
                v = self._Z[idx+1]
                if u == 0:
                    Z[self.m_storages, v-1] = 1
                elif v == 0:
                    Z[u-1,self.m_storages] = 1
                else:
                    Z[u-1, v-1] = 1
            Z[self._Z[len(self._Z)-1]-1, self.m_storages] = 1
            self._Z = Z
        
            