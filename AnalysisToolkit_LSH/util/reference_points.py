import numpy as np
from abc import ABC, abstractmethod
from scipy import special


class ReferenceDirectionFactory(ABC):

    def __init__(self, n_dim: int, scaling=None) -> None:
        self.n_dim = n_dim
        self.scaling = scaling

    def compute(self):
        if self.n_dim == 1:
            return np.array([[1.0]])
        else:
            ref_dirs = self._compute()
            if self.scaling is not None:
                ref_dirs = ref_dirs * self.scaling + ((1 - self.scaling) / self.n_dim)
            return ref_dirs

    @abstractmethod
    def _compute(self):
        pass


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(self, n_dim: int, scaling=None, n_points: int = None, n_partitions: int = None) -> None:
        super().__init__(n_dim, scaling)
        if n_points is not None:
            self.n_partitions = self.get_partition_closest_to_points(n_points, n_dim)
        else:
            if n_partitions is None:
                raise Exception("Either provide number of partitions or number of points.")
            else:
                self.n_partitions = n_partitions

    def _compute(self):
        return self.uniform_reference_directions(self.n_partitions, self.n_dim)

    def uniform_reference_directions(self, n_partitions: int, n_dim: int):
        ref_dirs = []
        ref_dir = np.full(n_dim, np.inf)
        self.__uniform_reference_directions(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

    def __uniform_reference_directions(self, ref_dirs, ref_dir, n_partitions: int, beta: int, depth: int):
        if depth == len(ref_dir) - 1:
            ref_dir[depth] = beta / (1.0 * n_partitions)
            ref_dirs.append(ref_dir[None, :])
        else:
            for i in range(beta + 1):
                ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
                self.__uniform_reference_directions(ref_dirs, np.copy(ref_dir), n_partitions, beta - i,
                                                    depth + 1)

    @staticmethod
    def get_partition_closest_to_points(n_points, n_dim):
        # in this case the do method will always return one values anyway
        if n_dim == 1:
            return 0

        n_partitions = 1
        _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)
        while _n_points <= n_points:
            n_partitions += 1
            _n_points = UniformReferenceDirectionFactory.get_n_points(n_partitions, n_dim)

        return n_partitions - 1

    @staticmethod
    def get_n_points(n_partitions, n_dim):
        return int(special.binom(n_dim + n_partitions - 1, n_partitions))
