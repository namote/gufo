# Older EXL3 conversion wrappers (XQ/F3)

Gufo-side helpers from the XQ and F3 conversion work, before the c03
pipeline in [`../flash_next/`](../flash_next/). No production receipt pins or
runs them; they are kept because earlier plan documents cite them.

| File | Purpose |
| --- | --- |
| `run_convert.py` | Runs the ExLlamaV3 converter in a GPU window and records time and memory |
| `provenance.py` | Writes PROVENANCE.json for an XQ pack |
| `check_pack.py` | Post-run checks of an XQ pack against the turboderp reference pack |
| `check_fn_pack.py` | Header and config checks of the downloaded turboderp Flash-Next EXL3 pack (XQ) |
| `compare_qtensors.py` | Compares converter qtensors (or a finished pack) with a reference pack by safetensors headers |
| `verify_hf_download.py` | Verifies a pinned Hugging Face download against the repository tree |

They were written for the XQ GPU windows and are not maintained.
