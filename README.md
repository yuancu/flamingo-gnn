# Flamingo for KG

In the original flamingo paper, the image token intertwines with the text token. However, in our setting, the *media* is not in a certain location of the text, but used to condition the whole generation process instead.

Therefore, this repository modifies the implementation of the Flamingo, and make it suitable for a more general encoder-decoder setting.
