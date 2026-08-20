"""Structural, execution, numerical, and metamorphic contracts."""

from .execution import (
    EVIDENCE_CONTRACT_SPECS,
    EvidenceContractId,
    EvidenceContractOutcome,
    EvidenceContractSpec,
    ExecutionEvidence,
    contiguous_stride,
    evaluate_execution_evidence,
    is_row_major_contiguous,
)
from .format import (
    FORMAT_CONTRACT_SPECS,
    ContractOutcome,
    ContractSpec,
    FormatContractId,
    evaluate_format_contracts,
)

__all__ = [
    "EVIDENCE_CONTRACT_SPECS",
    "FORMAT_CONTRACT_SPECS",
    "ContractOutcome",
    "ContractSpec",
    "EvidenceContractId",
    "EvidenceContractOutcome",
    "EvidenceContractSpec",
    "ExecutionEvidence",
    "FormatContractId",
    "contiguous_stride",
    "evaluate_execution_evidence",
    "evaluate_format_contracts",
    "is_row_major_contiguous",
]
