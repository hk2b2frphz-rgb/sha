"""Self-improving search loop for Whisper fine-tuning on the HPC GPU nodes.

The loop proposes a training/decoding configuration, trains a LoRA adapter,
scores it on a held-out dev set, and feeds the result back into the proposer.
It is designed to run unattended inside a single PBS job with a wall-clock
budget, checkpointing after every trial so a re-submitted job resumes.
"""
