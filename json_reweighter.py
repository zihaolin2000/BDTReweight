"""
Portable JSON serialization and inference for hep_ml GBReweighter models.

This module provides:

- ``export_reweighter_json``:
    Export a fitted ``hep_ml.reweight.GBReweighter``-compatible object
    to a portable JSON representation.

- ``JSONReweighter``:
    Load and evaluate the exported model without requiring ``hep_ml``
    or scikit-learn at inference time.

The input feature matrix is positional. Its columns must appear in the
same order used to train the original reweighter.

The evaluator reproduces scikit-learn tree traversal by converting each
input feature value to float32 while retaining tree thresholds as
float64.

Notes
-----
This module supports inference only. It does not train reweighters.

Non-finite input features are rejected intentionally. Although some
scikit-learn versions accept NaN values, their missing-value routing is
not exported by this format.
"""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "JSONReweighter",
    "export_reweighter_json",
    "load_json_reweighter",
]


MODEL_FORMAT = "hep_ml_gb_reweighter"
FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 262_144


class _CompiledTree(NamedTuple):
    """Dense, typed arrays used by the inference hot path."""

    feature: NDArray[np.int32]
    threshold: NDArray[np.float64]
    children_left: NDArray[np.int32]
    children_right: NDArray[np.int32]
    leaf_value: NDArray[np.float64]


def _as_path(path: str | Path) -> Path:
    """Return *path* as a ``Path`` object."""
    return path if isinstance(path, Path) else Path(path)


def export_reweighter_json(
    reweighter: Any,
    filepath: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """
    Export a fitted hep_ml GBReweighter-compatible object to JSON.

    The supplied object must expose a fitted ``.gb`` attribute whose
    ``estimators`` entries have the form::

        [tree, custom_leaf_values]

    This is the representation used by hep_ml's
    ``UGradientBoostingClassifier`` inside ``GBReweighter``.

    Parameters
    ----------
    reweighter
        A fitted reweighter object. This may be a direct
        ``hep_ml.reweight.GBReweighter`` instance or a subclass/wrapper
        that exposes the fitted model through ``reweighter.gb``.
    filepath
        Output JSON file.
    indent
        JSON indentation. Use ``None`` for compact output.

    Raises
    ------
    TypeError
        If the supplied object does not have the expected structure.
    RuntimeError
        If the model is not fitted or contains inconsistent tree data.
    ValueError
        If model metadata are invalid.
    """
    if not hasattr(reweighter, "gb"):
        raise TypeError(
            "Expected a fitted reweighter with a '.gb' attribute."
        )

    gb = reweighter.gb

    estimators = getattr(gb, "estimators", None)
    if estimators is None or len(estimators) == 0:
        raise RuntimeError(
            "The internal gradient-boosting model contains no estimators. "
            "Fit the reweighter before exporting it."
        )

    n_features = getattr(gb, "n_features", None)
    if n_features is None:
        n_features = getattr(reweighter, "n_features_", None)

    if n_features is None:
        raise RuntimeError(
            "Could not determine the number of input features."
        )

    n_features = int(n_features)
    if n_features <= 0:
        raise ValueError(
            f"Invalid number of features: {n_features}."
        )

    learning_rate = float(getattr(gb, "learning_rate"))
    initial_step = float(getattr(gb, "initial_step", 0.0))

    if not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite.")

    if not math.isfinite(initial_step):
        raise ValueError("initial_step must be finite.")

    model: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "format_version": FORMAT_VERSION,
        "n_features": n_features,
        "n_trees": len(estimators),
        "learning_rate": learning_rate,
        "initial_step": initial_step,
        "input_precision": "float32",
        "threshold_precision": "float64",
        "non_finite_policy": "reject",
        "trees": [],
    }

    for tree_index, estimator_entry in enumerate(estimators):
        if not isinstance(estimator_entry, (list, tuple)):
            raise TypeError(
                f"Estimator {tree_index} must be a list or tuple, "
                f"got {type(estimator_entry).__name__}."
            )

        if len(estimator_entry) != 2:
            raise RuntimeError(
                f"Estimator {tree_index} must contain exactly "
                "[tree, custom_leaf_values]."
            )

        tree, custom_values = estimator_entry

        if not hasattr(tree, "tree_"):
            raise TypeError(
                f"Estimator {tree_index} does not expose a '.tree_' "
                "attribute."
            )

        tree_data = tree.tree_
        node_count = int(tree_data.node_count)

        children_left = np.asarray(
            tree_data.children_left,
            dtype=np.int64,
        )
        children_right = np.asarray(
            tree_data.children_right,
            dtype=np.int64,
        )
        features = np.asarray(
            tree_data.feature,
            dtype=np.int64,
        )
        thresholds = np.asarray(
            tree_data.threshold,
            dtype=np.float64,
        )
        custom_values = np.asarray(
            custom_values,
            dtype=np.float64,
        ).reshape(-1)

        arrays = {
            "children_left": children_left,
            "children_right": children_right,
            "feature": features,
            "threshold": thresholds,
            "leaf_value": custom_values,
        }

        for name, array in arrays.items():
            if len(array) != node_count:
                raise RuntimeError(
                    f"Tree {tree_index}: '{name}' has length "
                    f"{len(array)}, expected {node_count}."
                )

        if not np.all(np.isfinite(thresholds[features >= 0])):
            raise RuntimeError(
                f"Tree {tree_index} contains a non-finite split threshold."
            )

        if not np.all(np.isfinite(custom_values)):
            raise RuntimeError(
                f"Tree {tree_index} contains a non-finite custom value."
            )

        nodes: list[dict[str, Any]] = []

        for node_id in range(node_count):
            left = int(children_left[node_id])
            right = int(children_right[node_id])
            is_leaf = left == -1 and right == -1

            if is_leaf:
                nodes.append(
                    {
                        "is_leaf": True,
                        "value": float(custom_values[node_id]),
                    }
                )
                continue

            if left < 0 or right < 0:
                raise RuntimeError(
                    f"Tree {tree_index}, node {node_id}: malformed child "
                    f"indices ({left}, {right})."
                )

            feature_index = int(features[node_id])
            if not 0 <= feature_index < n_features:
                raise RuntimeError(
                    f"Tree {tree_index}, node {node_id}: feature index "
                    f"{feature_index} is outside [0, {n_features})."
                )

            nodes.append(
                {
                    "is_leaf": False,
                    "feature": feature_index,
                    "threshold": float(thresholds[node_id]),
                    "left": left,
                    "right": right,
                }
            )

        model["trees"].append(
            {
                "node_count": node_count,
                "nodes": nodes,
            }
        )

    output_path = _as_path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            model,
            output_file,
            indent=indent,
            allow_nan=False,
        )
        output_file.write("\n")


class JSONReweighter:
    """
    Standalone evaluator for an exported hep_ml GBReweighter model.

    Parameters
    ----------
    model
        Parsed JSON model mapping.

    Notes
    -----
    Input matrices are positional. The number and ordering of columns
    must match the matrix used to train the original reweighter.
    """

    def __init__(self, model: Mapping[str, Any]) -> None:
        self._model = dict(model)
        self._validate_model()

        self.n_features = int(self._model["n_features"])
        self.n_trees = int(self._model["n_trees"])
        self.learning_rate = float(self._model["learning_rate"])
        self.initial_step = float(self._model["initial_step"])
        self._trees: list[dict[str, Any]] = list(self._model["trees"])
        self._compiled_trees = self._compile_trees()

    @classmethod
    def load(cls, filepath: str | Path) -> "JSONReweighter":
        """
        Load an exported reweighter from a JSON file.

        Parameters
        ----------
        filepath
            Path to the JSON model.

        Returns
        -------
        JSONReweighter
            Loaded standalone evaluator.
        """
        input_path = _as_path(filepath)

        with input_path.open("r", encoding="utf-8") as input_file:
            model = json.load(input_file)

        if not isinstance(model, Mapping):
            raise ValueError(
                "The top-level JSON value must be an object."
            )

        return cls(model)

    @property
    def model(self) -> Mapping[str, Any]:
        """Return the parsed model mapping."""
        return self._model

    def _validate_model(self) -> None:
        required_fields = {
            "format",
            "format_version",
            "n_features",
            "n_trees",
            "learning_rate",
            "initial_step",
            "trees",
        }

        missing = required_fields - self._model.keys()
        if missing:
            raise ValueError(
                f"JSON model is missing fields: {sorted(missing)}."
            )

        if self._model["format"] != MODEL_FORMAT:
            raise ValueError(
                f"Unsupported model format: {self._model['format']!r}."
            )

        version = int(self._model["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version {version}; "
                f"expected {FORMAT_VERSION}."
            )

        n_features = int(self._model["n_features"])
        n_trees = int(self._model["n_trees"])
        learning_rate = float(self._model["learning_rate"])
        initial_step = float(self._model["initial_step"])

        if n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {n_features}."
            )

        if n_trees < 0:
            raise ValueError(
                f"n_trees must be nonnegative, got {n_trees}."
            )

        if not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be finite.")

        if not math.isfinite(initial_step):
            raise ValueError("initial_step must be finite.")

        trees = self._model["trees"]
        if not isinstance(trees, list):
            raise ValueError("'trees' must be a list.")

        if len(trees) != n_trees:
            raise ValueError(
                f"n_trees is {n_trees}, but the model contains "
                f"{len(trees)} trees."
            )

        for tree_index, tree in enumerate(trees):
            if not isinstance(tree, Mapping):
                raise ValueError(
                    f"Tree {tree_index} must be a JSON object."
                )

            if "node_count" not in tree or "nodes" not in tree:
                raise ValueError(
                    f"Tree {tree_index} must contain 'node_count' "
                    "and 'nodes'."
                )

            node_count = int(tree["node_count"])
            nodes = tree["nodes"]

            if node_count <= 0:
                raise ValueError(
                    f"Tree {tree_index} has invalid node_count "
                    f"{node_count}."
                )

            if not isinstance(nodes, list):
                raise ValueError(
                    f"Tree {tree_index}: 'nodes' must be a list."
                )

            if len(nodes) != node_count:
                raise ValueError(
                    f"Tree {tree_index}: node_count is {node_count}, "
                    f"but {len(nodes)} nodes are present."
                )

            for node_id, node in enumerate(nodes):
                if not isinstance(node, Mapping):
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id} must be "
                        "a JSON object."
                    )

                if "is_leaf" not in node:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id} is missing "
                        "'is_leaf'."
                    )

                is_leaf = bool(node["is_leaf"])

                if is_leaf:
                    if "value" not in node:
                        raise ValueError(
                            f"Tree {tree_index}, leaf {node_id} is "
                            "missing 'value'."
                        )

                    value = float(node["value"])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Tree {tree_index}, leaf {node_id} has "
                            "a non-finite value."
                        )
                    continue

                split_fields = {
                    "feature",
                    "threshold",
                    "left",
                    "right",
                }
                missing_split_fields = split_fields - node.keys()

                if missing_split_fields:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id} is missing "
                        f"fields: {sorted(missing_split_fields)}."
                    )

                feature = int(node["feature"])
                threshold = float(node["threshold"])
                left = int(node["left"])
                right = int(node["right"])

                if not 0 <= feature < n_features:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id}: feature "
                        f"{feature} is outside [0, {n_features})."
                    )

                if not math.isfinite(threshold):
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id} has a "
                        "non-finite threshold."
                    )

                if not 0 <= left < node_count:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id}: invalid "
                        f"left child {left}."
                    )

                if not 0 <= right < node_count:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id}: invalid "
                        f"right child {right}."
                    )

                if left == node_id or right == node_id:
                    raise ValueError(
                        f"Tree {tree_index}, node {node_id} references "
                        "itself as a child."
                    )

    def _compile_trees(self) -> tuple[_CompiledTree, ...]:
        """Compile mapping-heavy JSON nodes into cache-friendly arrays."""
        compiled_trees: list[_CompiledTree] = []

        for tree in self._trees:
            nodes = tree["nodes"]
            node_count = int(tree["node_count"])

            feature = np.full(node_count, -1, dtype=np.int32)
            threshold = np.zeros(node_count, dtype=np.float64)
            children_left = np.full(node_count, -1, dtype=np.int32)
            children_right = np.full(node_count, -1, dtype=np.int32)
            leaf_value = np.zeros(node_count, dtype=np.float64)

            for node_id, node in enumerate(nodes):
                if bool(node["is_leaf"]):
                    leaf_value[node_id] = float(node["value"])
                else:
                    feature[node_id] = int(node["feature"])
                    threshold[node_id] = float(node["threshold"])
                    children_left[node_id] = int(node["left"])
                    children_right[node_id] = int(node["right"])

            for array in (
                feature,
                threshold,
                children_left,
                children_right,
                leaf_value,
            ):
                array.flags.writeable = False

            compiled_trees.append(
                _CompiledTree(
                    feature=feature,
                    threshold=threshold,
                    children_left=children_left,
                    children_right=children_right,
                    leaf_value=leaf_value,
                )
            )

        return tuple(compiled_trees)

    def _prepare_events(
        self,
        events: ArrayLike,
    ) -> NDArray[Any]:
        """Validate the shape and return a two-dimensional event matrix."""
        array = np.asarray(events)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                f"Expected a one- or two-dimensional input array, "
                f"got shape {array.shape}."
            )

        if array.shape[1] != self.n_features:
            raise ValueError(
                f"Model expects {self.n_features} columns, but the "
                f"input has {array.shape[1]}."
            )

        return array

    @staticmethod
    def _prepare_chunk(
        event_chunk: NDArray[Any],
        row_offset: int,
    ) -> NDArray[np.float32]:
        """Validate and convert one inference chunk to float32."""
        try:
            numeric_chunk = np.asarray(event_chunk, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Input features must be convertible to floating point."
            ) from error

        finite = np.isfinite(numeric_chunk)
        if not np.all(finite):
            flat_index = int(np.flatnonzero(~finite)[0])
            row, column = divmod(flat_index, numeric_chunk.shape[1])
            raise ValueError(
                f"Non-finite feature at row {row + row_offset}, "
                f"column {column}: {numeric_chunk[row, column]!r}."
            )

        # Values are intentionally converted before traversal. This matches
        # scikit-learn, which traverses float32 inputs against float64 splits.
        with np.errstate(over="ignore"):
            return np.asarray(numeric_chunk, dtype=np.float32)

    @staticmethod
    def _evaluate_tree_batch(
        tree: _CompiledTree,
        event_matrix: NDArray[np.float32],
        row_ids: NDArray[np.intp],
        node_ids: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        """
        Evaluate one tree for a batch of events.

        Only tree depth is traversed in Python. Each step routes every active
        event with NumPy advanced indexing, eliminating the Python loop over
        events used by the original evaluator.
        """
        node_ids.fill(0)
        active_rows = row_ids

        # A valid acyclic binary tree reaches a leaf in at most node_count
        # steps. The bound also prevents malformed multi-node cycles from
        # hanging inference indefinitely.
        for _ in range(tree.feature.size):
            active_nodes = node_ids[active_rows]
            active_features = tree.feature[active_nodes]
            is_split = active_features >= 0
            split_count = int(np.count_nonzero(is_split))

            if split_count == 0:
                return tree.leaf_value[node_ids]

            if split_count != is_split.size:
                active_rows = active_rows[is_split]
                active_nodes = active_nodes[is_split]
                active_features = active_features[is_split]

            feature_values = event_matrix[active_rows, active_features]
            go_left = feature_values <= tree.threshold[active_nodes]

            node_ids[active_rows] = np.where(
                go_left,
                tree.children_left[active_nodes],
                tree.children_right[active_nodes],
            )

        raise RuntimeError(
            "Tree traversal did not reach a leaf; the model may contain "
            "a cycle."
        )

    @staticmethod
    def _evaluate_tree_single(
        tree: _CompiledTree,
        event: NDArray[np.float32],
    ) -> float:
        """Evaluate one compiled tree without batch-allocation overhead."""
        node_id = 0

        for _ in range(tree.feature.size):
            feature_index = int(tree.feature[node_id])
            if feature_index < 0:
                return float(tree.leaf_value[node_id])

            if event[feature_index] <= tree.threshold[node_id]:
                node_id = int(tree.children_left[node_id])
            else:
                node_id = int(tree.children_right[node_id])

        raise RuntimeError(
            "Tree traversal did not reach a leaf; the model may contain "
            "a cycle."
        )

    def decision_function(
        self,
        events: ArrayLike,
        *,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> NDArray[np.float64]:
        """
        Return the ensemble score before exponentiation.

        Parameters
        ----------
        events
            One feature vector with shape ``(n_features,)`` or a matrix
            with shape ``(n_events, n_features)``.

        Returns
        -------
        numpy.ndarray
            One score per input event.

        Other Parameters
        ----------------
        chunk_size
            Maximum events evaluated together. Chunking bounds temporary
            traversal memory for very large samples. Pass ``None`` to process
            all events in one batch.
        """
        event_matrix = self._prepare_events(events)
        n_events = event_matrix.shape[0]

        if chunk_size is None:
            resolved_chunk_size = max(1, n_events)
        else:
            if isinstance(chunk_size, (bool, np.bool_)):
                raise TypeError("chunk_size must be a positive integer or None.")
            try:
                resolved_chunk_size = operator.index(chunk_size)
            except TypeError as error:
                raise TypeError(
                    "chunk_size must be a positive integer or None."
                ) from error
            if resolved_chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer.")

        scores = np.empty(n_events, dtype=np.float64)

        for start in range(0, n_events, resolved_chunk_size):
            stop = min(start + resolved_chunk_size, n_events)
            event_chunk = self._prepare_chunk(event_matrix[start:stop], start)
            chunk_scores = np.full(
                stop - start,
                self.initial_step,
                dtype=np.float64,
            )
            row_ids = np.arange(stop - start, dtype=np.intp)
            node_ids = np.empty(stop - start, dtype=np.int32)

            for tree in self._compiled_trees:
                contributions = self._evaluate_tree_batch(
                    tree,
                    event_chunk,
                    row_ids,
                    node_ids,
                )
                contributions *= self.learning_rate
                chunk_scores += contributions

            scores[start:stop] = chunk_scores

        return scores

    def predict_weights(
        self,
        events: ArrayLike,
        original_weight: ArrayLike | None = None,
        *,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    ) -> NDArray[np.float64]:
        """
        Predict reweighting factors or updated event weights.

        Parameters
        ----------
        events
            One feature vector or a two-dimensional feature matrix.
        original_weight
            Optional scalar or one-dimensional array of existing event
            weights. When omitted, only the learned multiplicative
            reweighting factors are returned.
        chunk_size
            Maximum events evaluated together. Pass ``None`` to process all
            events in one batch.

        Returns
        -------
        numpy.ndarray
            Predicted multiplicative factors, or factors multiplied by
            ``original_weight``.
        """
        multipliers = self.decision_function(events, chunk_size=chunk_size)
        np.exp(multipliers, out=multipliers)

        if original_weight is None:
            return multipliers

        weights = np.asarray(original_weight, dtype=np.float64)

        if weights.ndim == 0:
            if not np.isfinite(weights):
                raise ValueError("original_weight must be finite.")
            multipliers *= float(weights)
            return multipliers

        if weights.ndim != 1:
            raise ValueError(
                "original_weight must be a scalar or one-dimensional "
                "array."
            )

        if weights.shape != multipliers.shape:
            raise ValueError(
                f"original_weight has shape {weights.shape}, but "
                f"{multipliers.shape} was expected."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "original_weight contains a non-finite value."
            )

        multipliers *= weights
        return multipliers

    def predict_weight_single_event(
        self,
        features: ArrayLike,
        original_weight: float | None = None,
    ) -> float:
        """
        Predict the weight for one event.

        Parameters
        ----------
        features
            One feature vector with length ``n_features``.
        original_weight
            Optional existing event weight.

        Returns
        -------
        float
            Predicted multiplier, or updated weight when
            ``original_weight`` is supplied.
        """
        feature_array = np.asarray(features)

        if feature_array.ndim != 1:
            raise ValueError(
                f"Expected one feature vector, got shape "
                f"{feature_array.shape}."
            )

        if feature_array.shape[0] != self.n_features:
            raise ValueError(
                f"Model expects {self.n_features} features, but the "
                f"input has {feature_array.shape[0]}."
            )

        event = self._prepare_chunk(feature_array.reshape(1, -1), 0)[0]
        score = self.initial_step

        for tree in self._compiled_trees:
            score += self.learning_rate * self._evaluate_tree_single(
                tree,
                event,
            )

        multiplier = float(np.exp(np.float64(score)))

        if original_weight is None:
            return multiplier

        weight = float(original_weight)
        if not math.isfinite(weight):
            raise ValueError("original_weight must be finite.")

        return multiplier * weight


def load_json_reweighter(
    filepath: str | Path,
) -> JSONReweighter:
    """
    Load a JSON reweighter.

    This is a convenience wrapper around ``JSONReweighter.load``.

    Parameters
    ----------
    filepath
        Path to the JSON model.

    Returns
    -------
    JSONReweighter
        Loaded standalone evaluator.
    """
    return JSONReweighter.load(filepath)


def prepare_2p2h_reweight_inputs(
    nucleon1_px: ArrayLike, nucleon1_py: ArrayLike, nucleon1_pz: ArrayLike,
    nucleon2_px: ArrayLike, nucleon2_py: ArrayLike, nucleon2_pz: ArrayLike,
    lepton_px: ArrayLike, lepton_py: ArrayLike, lepton_pz: ArrayLike
) -> NDArray:
    """
    Prepare inputs for BDT reweight from SuSAv2 to Valencia exclusive 2p2h.
    Take lab frame arrays of px, py, pz of final state nucleon1, nucleon2,
    and lepton of length N.
    Return array of reaction plane frame array of shape N by 8, where columns
    are transformed px, py, pz of nucleon1 and nucleon2; py, pz of lepton.

    """

    lepton_pT = np.array([lepton_px, lepton_py]).T
    lepton_py_new = - np.linalg.norm(lepton_pT, axis=1)

    vector_y = - lepton_pT / (np.linalg.norm(lepton_pT, axis=1)[:,None]) # treat normalized lepton_py_new as -y
    vector_x = np.array([-vector_y[:,1], vector_y[:,0]]).T # vector x is rotating y clockwise by 90 deg
    
    nucleon1_pT = np.array([nucleon1_px, nucleon1_py]).T
    nucleon1_px_new = np.sum(nucleon1_pT * vector_x, axis = 1)
    nucleon1_py_new = np.sum(nucleon1_pT * vector_y, axis = 1)
    
    nucleon2_pT = np.array([nucleon2_px, nucleon2_py]).T
    nucleon2_px_new = np.sum(nucleon2_pT * vector_x, axis = 1)
    nucleon2_py_new = np.sum(nucleon2_pT * vector_y, axis = 1)


    return np.array([nucleon1_px_new, nucleon1_py_new, nucleon1_pz,
                     nucleon2_px_new, nucleon2_py_new, nucleon2_pz,
                     lepton_py_new, lepton_pz]).T


