# Flamingo for Encoder-Decoder

In the original flamingo paper, the image token intertwines with the text token. However, in our setting, the *media* is not in a certain location of the text, but used to condition the whole generation process instead.

Therefore, this repository modifies the implementation of the Flamingo, and make it suitable for a more general encoder-decoder setting.


## Encoder

Encoder is a t5 encoder based model, with GNN integrated into it.

Encoder related files:
- gnn.py
- t5_lmgnn.py

## Decoder

Decoder is a t5 decoder, but have extra cross attention heads to attend the GNN output from the encoder.

Decoder related files:
- gated_xattn.py
- flamingo_t5.py
- t5_decoder.py
