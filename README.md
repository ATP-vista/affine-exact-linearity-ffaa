# Reproducibility files: exact linearity over finite fields

This repository contains the computational supplement for Jinzhou Li,
*Exact linearity for affine configurations over finite fields* (manuscript).
It preserves the exact verification inputs and reference outputs used in the manuscript; it does not contain the manuscript itself.

## Requirements and reproduction

Python 3.10 or later is required. `separation.py` uses only the standard library; `verify.py` requires NumPy. Run with assertions enabled:

```sh
python3 separation.py > reproduced_separation.txt
diff -u separation_output.txt reproduced_separation.txt

python3 -m pip install numpy   # if NumPy is not already available
python3 verify.py > reproduced_verification.txt
diff -u verification.txt reproduced_verification.txt
```

`separation.py` checks the explicit rational bounds for the \(\mathbb F_7\) three-point code at block length 38 (25 symbols per port versus 26 for linear codes) and the \(\mathbb F_{31}\) five-point example. It includes deliberately failing tighter/shorter controls. `verify.py` checks the finite tables and characteristic-three circuit certificates, using `masses.json` for the integer input masses. The `.txt` files give the complete reference outputs. The manuscript provides the mathematical proofs and explains the precise scopes of the computations.

No licensed literature, unpublished manuscripts, correspondence or credentials are included.
