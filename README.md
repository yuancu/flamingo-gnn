# Flamingo for Encoder-Decoder

In the original flamingo paper, the image token intertwines with the text token. However, in our setting, the *media* is not in a certain location of the text, but used to condition the whole generation process instead.

Therefore, this repository modifies the implementation of the Flamingo, and make it suitable for a more general encoder-decoder setting.

## Dependencies
```
pip install -r requirements.txt
conda with conda install pyg -c pyg
```

## Architecture

There are two parallel data flows:
- A T5 model encodes and decodes language, it is kept intact during training.
- A GNN model encodes entity and relations, contextualized by the encoded language hidden states.

They are connected with two kinds of cross attentions:
- In the encoding stage, the context node in the graph corss attends language hidden states. Information flows from the language to the graph unilaterally.
- In the decoder, the hidden state at each decoding stage cross attends the graph hidden states, taking in the knowledge for decoding. Information flows back from the graph to the language.

### Encoder

Encoder is based on t5 encoder, where the last `k` hidden states are queried by GNN with cross attention.

Encoder related files:
- gnn.py
- t5_lmgnn.py

### Decoder

Decoder is based on t5 decoder. Extra cross attention layers are inserted in between original decoding blocks to attend the last GNN hidden state from the encoder.

Decoder related files:
- gated_xattn.py
- flamingo_t5.py
- t5_decoder.py
