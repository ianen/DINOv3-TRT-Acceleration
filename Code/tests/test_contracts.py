import numpy as np
import pytest

from dinov3_trt.contracts import (
    DINO_VITL16_224_CONTRACT,
    DINO_VITL16_LAYER_ABLATION_CANDIDATES,
    DINO_VITL16_LAYER_INDICES_DPT,
    DINO_VITL16_LAYER_INDICES_LATE,
    DINO_VITL16_LAYER_INDICES_PROJECT,
    DINO_VITL16_NUM_BLOCKS,
    derive_output_names,
    expected_output_shape,
    expected_token_count,
    make_dinov3_vitl16_contract,
    validate_output_shapes,
)


def test_dinov3_vitl16_main_contract_uses_trimmed_register_tokens() -> None:
    contract = DINO_VITL16_224_CONTRACT

    assert contract.layer_indices == (3, 11, 15, 19)
    assert contract.layer_numbers == (4, 12, 16, 20)
    assert contract.output_names == (
        "feat_layer_4",
        "feat_layer_12",
        "feat_layer_16",
        "feat_layer_20",
    )
    assert expected_token_count() == 197
    assert expected_token_count(include_register_tokens=True) == 201
    assert expected_output_shape(2) == (2, 197, 1024)


def test_make_dinov3_vitl16_contract_updates_static_resolution_token_count() -> None:
    contract_336 = make_dinov3_vitl16_contract(336)
    contract_518 = make_dinov3_vitl16_contract(518)

    assert contract_336.patch_grid == 21
    assert expected_token_count(contract_336) == 442
    assert expected_output_shape(2, contract_336) == (2, 442, 1024)
    assert contract_518.patch_grid == 32
    assert expected_token_count(contract_518) == 1025
    assert expected_token_count(contract_518, include_register_tokens=True) == 1029


def test_validate_output_shapes_accepts_all_four_outputs() -> None:
    outputs = {
        name: np.zeros((2, 197, 1024), dtype=np.float32)
        for name in DINO_VITL16_224_CONTRACT.output_names
    }

    validate_output_shapes(outputs, batch_size=2)


def test_validate_output_shapes_rejects_missing_output() -> None:
    outputs = {
        name: np.zeros((2, 197, 1024), dtype=np.float32)
        for name in DINO_VITL16_224_CONTRACT.output_names[:-1]
    }

    with pytest.raises(ValueError, match="missing"):
        validate_output_shapes(outputs, batch_size=2)


def test_validate_output_shapes_rejects_register_token_shape_on_main_path() -> None:
    outputs = {
        name: np.zeros((2, 201, 1024), dtype=np.float32)
        for name in DINO_VITL16_224_CONTRACT.output_names
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_output_shapes(outputs, batch_size=2)


def test_make_contract_accepts_layer_indices_override() -> None:
    contract = make_dinov3_vitl16_contract(224, layer_indices=DINO_VITL16_LAYER_INDICES_DPT)

    assert contract.layer_indices == (4, 10, 16, 22)
    assert contract.layer_numbers == (5, 11, 17, 23)
    assert contract.output_names == (
        "feat_layer_5",
        "feat_layer_11",
        "feat_layer_17",
        "feat_layer_23",
    )


def test_make_contract_accepts_layer_indices_late_variant() -> None:
    contract = make_dinov3_vitl16_contract(224, layer_indices=DINO_VITL16_LAYER_INDICES_LATE)

    assert contract.layer_numbers == (6, 12, 18, 24)
    assert contract.output_names[-1] == "feat_layer_24"


def test_make_contract_layer_indices_default_matches_project_main_path() -> None:
    overridden = make_dinov3_vitl16_contract(
        224, layer_indices=DINO_VITL16_LAYER_INDICES_PROJECT
    )

    assert overridden.layer_indices == DINO_VITL16_224_CONTRACT.layer_indices
    assert overridden.output_names == DINO_VITL16_224_CONTRACT.output_names


def test_derive_output_names_matches_one_based_layer_numbers() -> None:
    assert derive_output_names((0, 1, 2)) == ("feat_layer_1", "feat_layer_2", "feat_layer_3")
    assert derive_output_names((23,)) == ("feat_layer_24",)


def test_make_contract_rejects_layer_indices_out_of_range() -> None:
    with pytest.raises(ValueError, match="< 24"):
        make_dinov3_vitl16_contract(224, layer_indices=(0, 12, 23, 24))


def test_make_contract_rejects_negative_layer_indices() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_dinov3_vitl16_contract(224, layer_indices=(-1, 5, 10, 15))


def test_make_contract_rejects_unsorted_layer_indices() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_dinov3_vitl16_contract(224, layer_indices=(11, 3, 15, 19))


def test_make_contract_rejects_duplicate_layer_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_dinov3_vitl16_contract(224, layer_indices=(3, 11, 11, 19))


def test_make_contract_rejects_empty_layer_indices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_dinov3_vitl16_contract(224, layer_indices=())


def test_make_contract_rejects_non_int_layer_indices() -> None:
    with pytest.raises(TypeError, match="only int"):
        make_dinov3_vitl16_contract(224, layer_indices=(3.0, 11, 15, 19))  # type: ignore[arg-type]


def test_ablation_candidates_export_three_distinct_recipes() -> None:
    assert set(DINO_VITL16_LAYER_ABLATION_CANDIDATES) == {"project", "dpt", "late"}
    project = DINO_VITL16_LAYER_ABLATION_CANDIDATES["project"]
    dpt = DINO_VITL16_LAYER_ABLATION_CANDIDATES["dpt"]
    late = DINO_VITL16_LAYER_ABLATION_CANDIDATES["late"]
    assert project == (3, 11, 15, 19)
    assert dpt == (4, 10, 16, 22)
    assert late == (5, 11, 17, 23)
    assert DINO_VITL16_NUM_BLOCKS == 24
    for indices in (project, dpt, late):
        assert all(0 <= i < DINO_VITL16_NUM_BLOCKS for i in indices)
        assert list(indices) == sorted(indices)


def test_ablation_candidates_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        DINO_VITL16_LAYER_ABLATION_CANDIDATES["new"] = (0, 1, 2, 3)  # type: ignore[index]
