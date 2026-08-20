import torch
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize


def main() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 0)
    torch.manual_seed(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16) * 0.25
    a_global = (448 * 6) / a.float().abs().max()
    b_global = (448 * 6) / b.float().abs().max()

    with torch.cuda.nvtx.range("e001:nvfp4_quantize_a"):
        a_fp4, a_scale = nvfp4_quantize(
            a, a_global, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
    with torch.cuda.nvtx.range("e001:nvfp4_quantize_b"):
        b_fp4, b_scale = nvfp4_quantize(
            b, b_global, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )
    with torch.cuda.nvtx.range("e001:nvfp4_gemm"):
        output = mm_fp4(
            a_fp4,
            b_fp4.T,
            a_scale,
            b_scale.T,
            alpha=1.0 / (a_global * b_global),
            out_dtype=torch.bfloat16,
            use_nvfp4=True,
            backend="cutlass",
        )
        torch.cuda.synchronize()

    with torch.cuda.nvtx.range("e001:unquantized_reference"):
        error = (output.float() - a.float() @ b.float().T).abs()
        torch.cuda.synchronize()
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "shape": tuple(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "max_abs_error_vs_unquantized": float(error.max()),
        "mean_abs_error_vs_unquantized": float(error.mean()),
    }
    print("NVFP4_CUTLASS_OK", result)
    assert output.shape == (16, 128)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    main()
