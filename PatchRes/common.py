"""Nearest neighbour search utilities for anomaly scoring."""

import os
import pickle
from typing import List, Union

try:
    import faiss
except Exception:
    faiss = None
import numpy as np
from sklearn.neighbors import NearestNeighbors


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if faiss is None:
            raise ImportError("faiss is not available")
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options())
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        features = features.T
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        features = np.ascontiguousarray(features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(self, n_nearest_neighbours, query_features: np.ndarray,
            index_features: np.ndarray = None) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        if index_features is None:
            query_features = query_features.transpose(0, 1)
            query_features = np.ascontiguousarray(query_features)
            return self.search_index.search(query_features, n_nearest_neighbours)

        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class SklearnNN(object):
    def __init__(self, num_workers: int = 4) -> None:
        self.num_workers = num_workers
        self.nn = None

    def fit(self, features: np.ndarray) -> None:
        features = features.T
        features = np.ascontiguousarray(features)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=self.num_workers)
        self.nn.fit(features)

    def run(self, n_nearest_neighbours, query_features: np.ndarray,
            index_features: np.ndarray = None) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        if index_features is None:
            if self.nn is None:
                raise RuntimeError("Nearest neighbor index is not fitted")
            query_features = query_features.transpose(0, 1)
            query_features = np.ascontiguousarray(query_features)
            distances, indices = self.nn.kneighbors(query_features, n_neighbors=n_nearest_neighbours, return_distance=True)
            return distances, indices

        index_features = np.ascontiguousarray(index_features)
        query_features = np.ascontiguousarray(query_features)
        nn = NearestNeighbors(n_neighbors=n_nearest_neighbours, algorithm="auto", metric="euclidean", n_jobs=self.num_workers)
        nn.fit(index_features)
        distances, indices = nn.kneighbors(query_features, return_distance=True)
        return distances, indices

    def save(self, filename: str) -> None:
        raise NotImplementedError

    def load(self, filename: str) -> None:
        raise NotImplementedError

    def reset_index(self):
        self.nn = None


class _BaseMerger:
    def __init__(self):
        pass

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)


def _require_faiss_if_strict() -> None:
    flag = (os.environ.get("RES_SAM_REQUIRE_FAISS") or "").strip().lower()
    if flag in ("1", "true", "yes") and faiss is None:
        raise ImportError(
            "RES_SAM_REQUIRE_FAISS is set but faiss is not available."
        )


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=None) -> None:
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours

        _require_faiss_if_strict()
        if faiss is None:
            self.nn_method = SklearnNN(num_workers=4)
        else:
            self.nn_method = nn_method if nn_method is not None else FaissNN(False, 4)

        self.imagelevel_nn = lambda query: self.nn_method.run(n_nearest_neighbours, query)
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

    def predict(self, query_features: List[np.ndarray]) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features).T
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(self, save_folder: str, save_features_separately: bool = False, prepend: str = "") -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(self._detection_file(save_folder, prepend), self.detection_features)

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(self._detection_file(load_folder, prepend))
