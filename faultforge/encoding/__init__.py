"""Support for encoded **Systems**.

The functionality for encoded systems lives in :mod:`faultforge.encoding.system`.

Encoding Formats
----------------

The core definitions live in :mod:`faultforge.encoding.encoding`. An encoding
format will always come in two parts: an encoder and an encoding. The encoder is
responsible for encoding the data, while the encoding stores the encoded data
and is responsible for reconstructing the original data.

Primary formats:
- :mod:`faultforge.encoding.secded`: Hamming codes.
- :mod:`faultforge.encoding.bit_pattern`: Partial hamming codes.
- :mod:`faultforge.encoding.embedded_parity`: Embedding parity bits inside the data per chunk.
- :mod:`faultforge.encoding.mset`: Most Significant Exponent bit Triplicaiton.

Compositions:
- :mod:`faultforge.encoding.sequence`: Apply encodings sequentially.

As a convention, the ``*Encoding`` classes in these modules list the details
about the techniques.
"""
