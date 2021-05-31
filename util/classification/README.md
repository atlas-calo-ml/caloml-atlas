## Overview of files

`models.py` contains our "regular" classifiers.
`models_exp.py` contains our "experimental" classifiers. This is a good place to put models that may or may not be useful, to avoid cluttering up `models.py`.
`models_q.py` contains our *quantized* classifiers (quantized versions of the contents of `models.py`). Uses QKeras.