# Execution notes

The initial 2048-token embedding attempt used batches of eight on an 8 GiB WSL
host. The operating system killed it for memory use after the last logged count
of 9,600 questions. That attempt had no intermediate cache and contributed no
embeddings or outcomes to the reported results.

The 2048-token pass completed all 10,000 questions using batches of two, CPU
float32 and eight intra-operation threads. It took 8,466.7 seconds and peaked at
4,661.1 MiB RSS. Its head was selected on validation BCE and evaluated on the
separate 2,003-question test split. The 2048-token default and selected sample
were retained. Completed cache publication and progress checkpoints are atomic;
profile and ordered-question identity are checked before cache reuse.

The separate 512-token pass was stopped at the user's request to wrap up with
available data. Its saved checkpoint contains 6,400 questions processed in
ascending token length, all at most 199 tokens. No 512 head was trained or
evaluated. These short inputs are a length-biased subset; their 2,363.5-second
partial timing and partial peak memory cannot support a full-pass comparison.
The checkpoint remains local for an optional future experiment; it is not part
of the release bundle or a requirement for this handoff.

A tokenizer-only pass over all 10,000 selected prompts found 1,410 inputs above
512 total tokens and none above 2048. All 6,400 completed short-input embeddings
match the completed 2048 cache exactly. `partial-512.json` records these results;
`scripts/summarize_partial_success_evaluation.py` reproduces the summary from the
source extracts, pinned tokenizer, completed cache and interrupted checkpoint.

Reported wall time and memory exclude the discarded attempt and later reads of
the completed cache. Timing occurred during development on a shared host and is
not a controlled hardware benchmark. No relative accuracy, calibration, routed
cost, full latency or full memory claim is made for 512 versus 2048.
