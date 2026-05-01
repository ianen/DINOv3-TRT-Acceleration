import numpy as np
import pytest
from numpy.typing import NDArray

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT
from dinov3_trt.export.wrapper import (
    DinoV3IntermediateLayerWrapper,
    ordered_output_mapping,
    prepend_class_token,
)


class FakeDinoModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.seen_layers: tuple[int, ...] | None = None
        self.seen_return_class_token: bool | None = None

    def eval(self) -> None:
        self.eval_called = True

    def get_intermediate_layers(
        self,
        pixel_values: NDArray[np.float32],
        n: tuple[int, ...],
        return_class_token: bool = False,
    ) -> tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...] | tuple[NDArray[np.float32], ...]:
        self.seen_layers = n
        self.seen_return_class_token = return_class_token
        batch_size = pixel_values.shape[0]
        patch_tokens = tuple(np.zeros((batch_size, 196, 1024), dtype=np.float32) for _ in n)
        if not return_class_token:
            return patch_tokens
        class_token = np.ones((batch_size, 1024), dtype=np.float32)
        return tuple((patch_tokens_for_layer, class_token) for patch_tokens_for_layer in patch_tokens)


def test_wrapper_calls_project_layer_indices_and_returns_tuple() -> None:
    model = FakeDinoModel()
    wrapper = DinoV3IntermediateLayerWrapper(model).eval()
    outputs = wrapper(np.zeros((2, 3, 224, 224), dtype=np.float32))

    assert model.eval_called is True
    assert model.seen_layers == DINO_VITL16_224_CONTRACT.layer_indices
    assert model.seen_return_class_token is True
    assert isinstance(outputs, tuple)
    assert len(outputs) == 4
    assert outputs[0].shape == (2, 197, 1024)
    np.testing.assert_array_equal(outputs[0][:, 0, :], np.ones((2, 1024), dtype=np.float32))


def test_prepend_class_token_accepts_numpy_arrays() -> None:
    patch_tokens = np.zeros((1, 196, 1024), dtype=np.float32)
    class_token = np.ones((1, 1024), dtype=np.float32)

    output = prepend_class_token(patch_tokens, class_token)

    assert output.shape == (1, 197, 1024)
    np.testing.assert_array_equal(output[:, 0, :], class_token)


def test_ordered_output_mapping_checks_cardinality() -> None:
    with pytest.raises(ValueError, match="does not match"):
        ordered_output_mapping(("a", "b"), (np.zeros((1,)),))


def test_wrapper_rejects_models_without_intermediate_api() -> None:
    with pytest.raises(TypeError, match="get_intermediate_layers"):
        DinoV3IntermediateLayerWrapper(object())
