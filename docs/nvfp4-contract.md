# NVFP4 Format Contract

This document defines the Gate 1 format oracle for dense, row-wise NVFP4 GEMM
operands. It is derived from the pinned public sources in
`experiments/E002-format-oracle/sources.json`, not from FlashInfer's encoder,
decoder, or CUDA kernels.

## Domain and terminology

- A logical matrix has positive shape `(rows, columns)`.
- `columns` is divisible by the dense NVFP4 block size of 16 and by the E2M1
  packing width of two.
- Each logical row owns `columns / 16` local scale codes.
- The selected execution layout is CUTLASS `128x4`, the layout used by E001.
  Linear scale storage is the canonical logical form used by the oracle.
- `global_scale` means the FP32 multiplier used during dequantization.
  FlashInfer's quantizer argument is its reciprocal, so the two names must not
  be interchanged.

## E2M1 values

The four-bit payload contains a sign bit and a three-bit magnitude payload.
The exact positive table is:

| Magnitude code | Value |
| --- | ---: |
| `000` | 0 |
| `001` | 0.5 |
| `010` | 1 |
| `011` | 1.5 |
| `100` | 2 |
| `101` | 3 |
| `110` | 4 |
| `111` | 6 |

The high bit negates the value, including a distinct negative-zero payload.
There are no infinity or NaN encodings. Conversion in the oracle accepts only
finite inputs, saturates magnitudes above six, and resolves ties to the code
whose least-significant payload bit is zero.

## Packed values

CUTLASS sub-byte memory references map logical element offset zero to bit
offset zero. Therefore, for this repository's tensor-storage contract, logical
element `2i` occupies the low nibble and logical element `2i+1` occupies the
high nibble. PTX conversion-instruction operand ordering is a separate concern
and must not be used to reverse the memory convention.

Packing or unpacking an odd logical element count is invalid. No implicit
padding is permitted for packed values.

## UE4M3 local scales

UE4M3 is an unsigned seven-bit payload stored in a byte whose most-significant
bit must be zero. It has exponent bias seven, three mantissa bits, finite range
zero through 448, and a NaN code at `0x7f`. Scale inputs reject the NaN code and
all bytes with a set padded MSB.

For exponent `e` and mantissa `m`:

```text
e == 0: value = m * 2^-9
e != 0: value = (1 + m/8) * 2^(e-7)
```

The E002 oracle covers all 127 finite codes. Encoding is non-negative,
saturating, round-to-nearest-even conversion.

## Hierarchical reconstruction

For logical element `(row, column)`:

```text
scale_column = floor(column / 16)
x[row, column] =
    decode_e2m1(value_code[row, column])
    * decode_ue4m3(scale_code[row, scale_column])
    * global_scale
```

`global_scale` is represented explicitly as a scalar field rather than a fake
one-element or zero-dimensional tensor shape. This resolves the schema-v1
ambiguity found in E001.

The public quantization recipe defines:

```text
global_scale = global_amax / (448 * 6)
raw_block_scale = (block_amax / 6) / global_scale
```

An all-zero tensor produces a zero global scale and zero block scales in the
mathematical oracle. Adapter-specific reciprocal-scale handling remains an
adapter precondition.

## CUTLASS 128x4 scale layout

The physical layout pads rows to a multiple of 128 and scale columns to a
multiple of four. Every `128 x 4` atom occupies 512 bytes. For logical
`(row, scale_column)`:

```text
row_atom       = row // 128
row_in_atom    = row % 128
row_group      = row_in_atom // 32
row_in_group   = row_in_atom % 32
scale_atom     = scale_column // 4
scale_in_atom  = scale_column % 4
scale_atoms    = ceil(logical_scale_columns / 4)

atom_base = (row_atom * scale_atoms + scale_atom) * 512
offset = atom_base + row_in_group * 16 + row_group * 4 + scale_in_atom
```

Padding is physical storage, not part of the logical scale matrix. Conversion
between linear and CUTLASS layouts is explicit and reversible for logical
entries.

## Contract limits

- This contract covers dense row-wise block-16 operands and CUTLASS `128x4`.
- Transformer Engine's two-dimensional training scale recipe is recorded but
  is not silently treated as this GEMM operand layout.
- FlashInfer-specific `8x4`, post-quantization shuffle, per-token scaling, and
  checkpoint-specific permutations are outside this Gate 1 contract.
- Exact format agreement does not establish numerical-error bounds, GEMM
  correctness, model-level quality, or performance. Those belong to later
  gates.
