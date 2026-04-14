"""Export (or download) SDXL component models to ONNX.

Provides a single entry point :func:`ensure_models` that prepares all ONNX
models required for a given precision mode.  Individual component functions
are also usable standalone:

- :func:`export_text_encoders` — CLIP-L + OpenCLIP-bigG
- :func:`export_unet` — fp32/fp16 from diffusers, int8 downloaded from Azure
- :func:`export_vae` — VAE decoder from ``madebyollin/sdxl-vae-fp16-fix``
"""

import logging
import pathlib
import urllib.request

logger = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

_INT8_UNET_AZURE_BASE = (
    "https://sharkpublic.blob.core.windows.net/sharkpublic" "/SDXL/ONNX/unet/int8"
)
_INT8_UNET_FILES = [
    "sdxl_int8_unet_native_i8.onnx",
    "sdxl_int8_unet_native_i8.onnx.data",
]


# ---------------------------------------------------------------------------
# Text Encoders
# ---------------------------------------------------------------------------


def export_text_encoders(models_dir: pathlib.Path, dtype: str) -> None:
    """Export both CLIP text encoders from HuggingFace.

    Args:
        models_dir: Root directory for cached models.
        dtype: ``"fp32"`` or ``"fp16"``.
    """
    import torch
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    dummy_ids = torch.randint(0, 49408, (1, 77))

    # --- Text Encoder 1 (CLIP-L) ---
    out_dir1 = models_dir / f"text_encoder_1_{dtype}"
    if not (out_dir1 / "model.onnx").exists():
        out_dir1.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting Text Encoder 1 (%s)...", dtype)

        class TE1(torch.nn.Module):
            def __init__(self, te):
                super().__init__()
                self.te = te

            def forward(self, input_ids):
                return self.te(input_ids, output_hidden_states=True).hidden_states[-2]

        te1 = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=torch_dtype
        )
        te1.eval()
        torch.onnx.export(
            TE1(te1).eval(),
            (dummy_ids,),
            str(out_dir1 / "model.onnx"),
            input_names=["input_ids"],
            output_names=["hidden_states"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=True,
            external_data=False,
        )
        logger.info("  Saved: %s", out_dir1 / "model.onnx")
        del te1
    else:
        logger.info("Text Encoder 1 already exported: %s", out_dir1)

    # --- Text Encoder 2 (OpenCLIP-bigG) ---
    out_dir2 = models_dir / f"text_encoder_2_{dtype}"
    if not (out_dir2 / "model.onnx").exists():
        out_dir2.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting Text Encoder 2 (%s)...", dtype)

        class TE2(torch.nn.Module):
            def __init__(self, te):
                super().__init__()
                self.te = te

            def forward(self, input_ids):
                o = self.te(input_ids, output_hidden_states=True)
                return o.hidden_states[-2], o.text_embeds

        te2 = CLIPTextModelWithProjection.from_pretrained(
            MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch_dtype
        )
        te2.eval()
        torch.onnx.export(
            TE2(te2).eval(),
            (dummy_ids,),
            str(out_dir2 / "model.onnx"),
            input_names=["input_ids"],
            output_names=["hidden_states", "text_embeds"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=True,
            external_data=True,
        )
        logger.info("  Saved: %s", out_dir2 / "model.onnx")
        del te2
    else:
        logger.info("Text Encoder 2 already exported: %s", out_dir2)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


def _download_int8_unet(models_dir: pathlib.Path) -> None:
    """Download pre-quantized W8A8 int8 UNet from Azure sharkpublic."""
    out_dir = models_dir / "unet_int8"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in _INT8_UNET_FILES:
        dest = out_dir / fname
        if dest.exists():
            logger.info("  Already exists: %s", dest)
            continue
        url = f"{_INT8_UNET_AZURE_BASE}/{fname}"
        logger.info("  Downloading %s ...", fname)
        urllib.request.urlretrieve(url, dest)
        logger.info("  Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def export_unet(models_dir: pathlib.Path, dtype: str) -> None:
    """Export (or download) SDXL UNet.

    For ``fp32`` / ``fp16``: exports from HuggingFace diffusers.
    For ``int8``: downloads the pre-quantized W8A8 ONNX from Azure.

    Args:
        models_dir: Root directory for cached models.
        dtype: ``"fp32"``, ``"fp16"``, or ``"int8"``.
    """
    if dtype == "int8":
        _download_int8_unet(models_dir)
        return

    import torch
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    from diffusers import UNet2DConditionModel

    from .onnx_utils import (
        inline_slice_constants,
        patch_resize_ops,
        fix_fp16_type_mismatches,
    )

    out_dir = models_dir / f"unet_{dtype}"
    if (out_dir / "model.onnx").exists():
        logger.info("UNet already exported: %s", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    logger.info("Exporting UNet (%s) — this takes a few minutes...", dtype)

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch_dtype
    )
    unet.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(
            self, latent_sample, timestep, encoder_hidden_states, text_embeds, time_ids
        ):
            return self.unet(
                latent_sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    out_path = str(out_dir / "model.onnx")
    prog = torch.onnx.export(
        Wrapper(unet).eval(),
        (
            torch.randn(1, 4, 128, 128, dtype=torch_dtype),
            torch.tensor([250], dtype=torch_dtype),
            torch.randn(1, 77, 2048, dtype=torch_dtype),
            torch.randn(1, 1280, dtype=torch_dtype),
            torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch_dtype),
        ),
        input_names=[
            "latent_sample",
            "timestep",
            "encoder_hidden_states",
            "text_embeds",
            "time_ids",
        ],
        output_names=["sample"],
        dynamo=True,
        external_data=True,
    )
    prog.save(out_path)

    model = onnx.load(out_path, load_external_data=True)
    model = inline_slice_constants(model)
    model = patch_resize_ops(model)
    if dtype == "fp16":
        model = fix_fp16_type_mismatches(model)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    onnx.save(model, out_path)
    logger.info("  Saved: %s", out_path)
    del unet


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def export_vae(models_dir: pathlib.Path, dtype: str) -> None:
    """Export SDXL VAE decoder from HuggingFace.

    Args:
        models_dir: Root directory for cached models.
        dtype: ``"fp32"`` or ``"fp16"``.
    """
    import torch
    import onnx
    from diffusers import AutoencoderKL

    from .onnx_utils import patch_resize_ops

    out_dir = models_dir / f"vae_decoder_{dtype}"
    if (out_dir / "model.onnx").exists():
        logger.info("VAE already exported: %s", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    logger.info("Exporting VAE decoder (%s)...", dtype)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
    )
    vae.eval()

    class Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            return self.vae.decode(latent_sample, return_dict=False)[0]

    out_path = str(out_dir / "model.onnx")
    torch.onnx.export(
        Decoder(vae).eval(),
        (torch.randn(1, 4, 128, 128, dtype=torch_dtype),),
        out_path,
        input_names=["latent_sample"],
        output_names=["sample"],
        opset_version=18,
        do_constant_folding=True,
    )

    model = onnx.load(out_path)
    model = patch_resize_ops(model)
    onnx.save(model, out_path)
    logger.info("  Saved: %s", out_path)
    del vae


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def ensure_models(models_dir: pathlib.Path, dtype: str) -> None:
    """Export or download all models needed for the given dtype.

    Args:
        models_dir: Root directory for cached models.
        dtype: ``"fp32"``, ``"fp16"``, or ``"int8"``.
              ``int8`` uses fp16 text encoders and VAE with a downloaded
              W8A8 quantized UNet.
    """
    te_dtype = "fp16" if dtype == "int8" else dtype
    vae_dtype = "fp16" if dtype == "int8" else dtype

    logger.info("=== Preparing models (dtype=%s) ===", dtype)
    export_text_encoders(models_dir, te_dtype)
    export_unet(models_dir, dtype)
    export_vae(models_dir, vae_dtype)
    logger.info("=== All models ready ===\n")
