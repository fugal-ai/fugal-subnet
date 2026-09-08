# Execution notes

The initial 2048-token embedding attempt used batches of eight on an 8 GiB WSL
host. The operating system killed it for memory use after the last logged count
of 9,600 questions. That attempt had no intermediate cache and contributed no
embeddings or outcomes to the reported comparison.

The complete comparison was restarted with batches of two. Both input limits
use that batch size, CPU float32, and eight intra-operation threads. Embedding
progress is now saved atomically every 200 questions, with the profile and
ordered-question identity checked before a restart can reuse it. The original
2048-token default and the selected sample were retained.

The reported wall times and peak resident memory describe the successful
embedding passes. They exclude the discarded attempt and subsequent reads of
the completed caches. Timing occurred during development on a shared host and
is not a controlled hardware benchmark. Reproduction on another host can have
different timings and memory use.
