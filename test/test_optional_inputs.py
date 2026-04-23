"""Test that absent/optional ONNX inputs emit torch.constant.none in MLIR."""

import pathlib
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array

# Fixed seed for reproducibility.
np.random.seed(42)


def _make_clip_absent_min_model():
    """Build Clip(X, min="", max) -- 1 absent middle input."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    max_init = from_array(np.array(2.0, dtype=np.float32), name="max")

    clip = helper.make_node("Clip", inputs=["X", "", "max"], outputs=["Y"])

    graph = helper.make_graph([clip], "clip_test", [X], [Y], initializer=[max_init])
    model = helper.make_model(
        graph,
        producer_name="optional_inputs_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    return model


@pytest.fixture(params=["cpu", "gpu"])
def iree_device_and_target(request):
    """Yield (device, target_arch) for each backend."""
    if request.param == "cpu":
        return request.getfixturevalue("iree_device"), "host"
    return (
        request.getfixturevalue("iree_gpu_device"),
        request.getfixturevalue("gpu_target"),
    )


def test_clip_absent_min_e2e(iree_device_and_target):
    """Clip(X, min="", max) compiles and produces correct output."""
    device, target_arch = iree_device_and_target
    model = _make_clip_absent_min_model()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        opts = ort.SessionOptions()
        opts.add_provider_for_devices([device], {"target_arch": target_arch})
        sess = ort.InferenceSession(model_path, sess_options=opts)

        x = np.array([-3.0, 0.0, 4.0, 7.0], dtype=np.float32)
        result = sess.run(None, {"X": x})[0]

        expected = np.array([-3.0, 0.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    finally:
        pathlib.Path(model_path).unlink(missing_ok=True)
