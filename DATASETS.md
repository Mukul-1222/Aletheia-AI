# FINAL25 data and corpus notes

FINAL25 does not ship a pretrained tokenizer or model.

## Tokenizer training

The learned BPE vocabulary is generated only from text/data explicitly supplied to `--tokenizer-train` or `AletheiaAgent.train_tokenizer()`.

The tokenizer always retains byte fallback, so any UTF-8 string remains representable exactly.

## Language evaluation

The fixed evaluation corpus remains local to the Aletheia root. Training paths are filtered from the evaluation corpus according to the existing evaluation builder rules.

The model metric is true bits per UTF-8 byte, making byte-token and learned-BPE checkpoints comparable.

## Open-world outcome sources

Open-world verification derives signals from observations already available to Aletheia: predicted latent observations, previously stored source representations, and optional self-generated source-grounded questions. No manually authored world reward table is required.
