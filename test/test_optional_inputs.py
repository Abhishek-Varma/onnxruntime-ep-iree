"""Test that absent/optional ONNX inputs emit torch.constant.none in MLIR."""

import re

import numpy as np
from conftest import try_generate_mlir
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array


def _make_clip_model(input_names):
    """Build a Clip model with the given input names.

    Pass an empty string in `input_names` to mark that positional input as
    absent (ONNX's representation of an unsupplied optional input).  Bound
    inputs ("min" and/or "max") are emitted as scalar f32 initializers.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])

    initializers = []
    if "min" in input_names:
        initializers.append(from_array(np.array(-1.0, dtype=np.float32), name="min"))
    if "max" in input_names:
        initializers.append(from_array(np.array(2.0, dtype=np.float32), name="max"))

    clip = helper.make_node("Clip", inputs=list(input_names), outputs=["Y"])
    graph = helper.make_graph([clip], "clip_test", [X], [Y], initializer=initializers)
    model = helper.make_model(
        graph,
        producer_name="optional_inputs_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


def _find_clip_op_line(mlir):
    for line in mlir.splitlines():
        if 'torch.operator "onnx.Clip"' in line:
            return line.strip()
    raise AssertionError(f"onnx.Clip op not found in generated MLIR:\n{mlir}")


def test_absent_input_emits_none_and_preserves_position(cpu_device, target_arch):
    """Clip(X, "", max): %__none must be emitted and referenced in slot 1."""
    model = _make_clip_model(["X", "", "max"])
    mlir, err = try_generate_mlir(
        model, cpu_device, "", target_arch, assert_compiles=True
    )
    assert err is None, f"MLIR generation failed: {err}"

    # The function-scoped %__none constant must be emitted exactly once.
    assert "%__none = torch.constant.none" in mlir, mlir

    # The Clip op must reference %__none in the second operand slot, between
    # the X tensor and the max initializer.
    op_line = _find_clip_op_line(mlir)
    assert re.search(r"\(%X, %__none, %max\)", op_line), op_line


def test_multiple_absent_inputs_emit_none_once(cpu_device, target_arch):
    """Clip(X, "", "") emits a single %__none reused by both absent slots.s"""
    model = _make_clip_model(["X", "", ""])
    mlir, err = try_generate_mlir(
        model, cpu_device, "", target_arch, assert_compiles=True
    )
    assert err is None, f"MLIR generation failed: {err}"

    # Exactly one definition of %__none in the function.
    assert mlir.count("%__none = torch.constant.none") == 1, mlir

    op_line = _find_clip_op_line(mlir)
    assert re.search(r"\(%X, %__none, %__none\)", op_line), op_line
