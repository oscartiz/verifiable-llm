# Paper

`verifiable-llm.tex` — a short applied-cryptography / systems paper describing
the project: the provenance problem, the three-layer protocol, the
bounded-drift self-attack and its deterministic fix, the two zero-knowledge
decode-proof backends, and an evaluation on an Apple M4.

## Build

The source uses only ubiquitous packages (`amsmath`, `booktabs`, `hyperref`, …)
and is pure ASCII, so it builds anywhere:

```sh
# local (MacTeX / TeX Live)
pdflatex verifiable-llm.tex && pdflatex verifiable-llm.tex   # twice for refs

# or paste verifiable-llm.tex into https://overleaf.com (compiler: pdfLaTeX)
```

Set your name in the `\author{...}` line before submitting (there is a `% TODO`
marker). The numbers are grounded in [`../BENCHMARKS.md`](../BENCHMARKS.md);
re-run those benches to refresh them for a different machine.
