"""ONNX post-processing helpers for IREE compatibility.

These transforms fix issues in torch.onnx.export output that prevent
correct lowering through the torch-mlir ONNX-to-Torch pipeline used by
the IREE Execution Provider.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def patch_resize_ops(model):
    """Patch Resize ops to 4-input format (X, roi, scales, sizes) for IREE.

    The dynamo exporter produces ``Resize(X, "", scales)`` with an empty-string
    roi input.  The EP's MLIR generator skips empty-name inputs, producing only
    2 inputs in MLIR.  IREE needs all 4 positional inputs.
    """
    from onnx import numpy_helper

    roi_name = "_resize_roi_empty"
    roi_tensor = numpy_helper.from_array(np.array([], dtype=np.float32), name=roi_name)
    model.graph.initializer.append(roi_tensor)
    count = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        count += 1
        inputs = list(node.input)
        while len(inputs) < 4:
            inputs.append("")
        if inputs[1] == "":
            inputs[1] = roi_name
        del node.input[:]
        node.input.extend(inputs)
    if count:
        logger.info("  Patched %d Resize ops", count)
    return model


def inline_slice_constants(model):
    """Inline Slice initializer refs as Constant ops for torch-mlir.

    torch-mlir's ONNX importer expects Slice parameters (starts, ends, axes,
    steps) as ``onnx.Constant`` ops, not graph initializers.
    """
    from onnx import helper

    init_map = {i.name: i for i in model.graph.initializer}
    new_nodes, remap = [], {}
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        for inp in list(node.input)[1:]:
            if not inp or inp in remap or inp not in init_map:
                continue
            new_name = f"_inlined_{inp}"
            c = helper.make_node("Constant", [], [new_name])
            c.attribute.append(helper.make_attribute("value", init_map[inp]))
            new_nodes.append(c)
            remap[inp] = new_name
    if not new_nodes:
        return model
    for node in model.graph.node:
        if node.op_type != "Slice":
            continue
        for i in range(1, len(node.input)):
            if node.input[i] in remap:
                node.input[i] = remap[node.input[i]]
    for c in reversed(new_nodes):
        model.graph.node.insert(0, c)
    logger.info("  Inlined %d Slice constants", len(new_nodes))
    return model


def fix_fp16_type_mismatches(model):
    """Insert Cast ops where fp16/fp32 inputs are mixed in same-type ops.

    The dynamo exporter sometimes leaves scalar fp32 constants feeding into
    otherwise-fp16 graphs.  Ops like ``Add``, ``Concat``, ``MatMul`` require
    uniform element types.
    """
    import onnx
    from onnx import helper

    type_map = {}
    for inp in model.graph.input:
        type_map[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_map[init.name] = init.data_type
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    for out in node.output:
                        type_map[out] = attr.i
        else:
            it = next((type_map[i] for i in node.input if i and i in type_map), None)
            if it is not None:
                for out in node.output:
                    type_map[out] = it

    same_type_ops = {"Concat", "Add", "Sub", "Mul", "Div", "MatMul", "Gemm"}
    fix_count = 0
    inserts = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in same_type_ops:
            continue
        types = {type_map.get(i) for i in node.input if i and i in type_map}
        types.discard(None)
        if len(types) <= 1:
            continue
        for i, inp in enumerate(node.input):
            if inp and type_map.get(inp) == onnx.TensorProto.FLOAT:
                co = f"_fixcast_{fix_count}_{inp}"
                cn = helper.make_node("Cast", [inp], [co], to=onnx.TensorProto.FLOAT16)
                node.input[i] = co
                type_map[co] = onnx.TensorProto.FLOAT16
                inserts.append((idx, cn))
                fix_count += 1
    for idx, cn in reversed(inserts):
        model.graph.node.insert(idx, cn)
    if fix_count:
        logger.info("  Fixed %d fp16/fp32 type mismatches", fix_count)
    return model
