# Vendored IFEval provenance

`instructions.py`, `instructions_registry.py`, and the base implementation of
`instructions_util.py` are vendored from Google Research commit
`26d8ccdab6fec61b5c83ad6327ea8bda9e580288` under Apache-2.0. Fugal changes
only package imports, deterministic NLTK resource loading, and the call site of
the equivalent word-tokenization helper. `evaluator.py` is Fugal's strict,
fail-closed adapter around those classes.

`punkt-english.tar.gz.b64` contains only the English `punkt_tab` parameters
from `nltk/nltk_data` commit
`550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`. Its decoded SHA-256 is
`94cf7aa3b7ca8645047a43058e62e4884d7872e0f671b6e6f56e5716ea0a66ef`.

Upstream license notices remain at the top of each Google source file. See the
repository `NOTICE` file for third-party attribution and license links.
