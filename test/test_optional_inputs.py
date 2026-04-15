"""Test that absent/optional ONNX inputs emit torch.constant.none in MLIR."""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array


def _make_resize_model(use_sizes=False):
    """Build a nearest 2x Resize with absent optional inputs.

    Args:
        use_sizes: If False, emit Resize(X, roi="", scales)   — 1 absent input.
                   If True,  emit Resize(X, roi="", scales="", sizes) — 2 absent.
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 16, 16])

    if use_sizes:
        init = from_array(np.array([1, 4, 16, 16], dtype=np.int64), name="sizes")
        inputs = ["X", "", "", "sizes"]
    else:
        init = from_array(
            np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
        )
        inputs = ["X", "", "scales"]

    resize = helper.make_node(
        "Resize",
        inputs=inputs,
        outputs=["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )

    graph = helper.make_graph([resize], "resize_test", [X], [Y], initializer=[init])
    model = helper.make_model(
        graph,
        producer_name="optional_inputs_test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    return model


@pytest.mark.gpu
@pytest.mark.parametrize("use_sizes", [False, True], ids=["scales", "sizes"])
def test_resize_absent_inputs_e2e(iree_gpu_device, gpu_target, use_sizes):
    """Resize with absent optional inputs compiles and produces correct output."""
    model = _make_resize_model(use_sizes=use_sizes)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([iree_gpu_device], {"target_arch": gpu_target})
        sess = ort.InferenceSession(model_path, sess_options=opts)

        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        result = sess.run(None, {"X": x})[0]

        expected = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)
