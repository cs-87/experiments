"""
Which bit of the payload a given frame carries.

Its own module because it is the one thing the embedder and every detector must agree
on exactly: an off-by-one here is not a degradation, it is a different payload. Anything
that writes or reads the mark calls bit_index_for_frame() rather than re-deriving the
expression, which is how the two sides stayed out of step before.

Two maps, and the difference between them is the whole of finding F4.

  contiguous:  bit_index = (i // temp_redundancy) % bit_length
  interleaved: bit_index = i % bit_length

Contiguous holds one bit still for temp_redundancy consecutive frames. Those frames are
one scene measured repeatedly, not independent draws -- measured per-frame error
autocorrelation 0.69 at lag 1 and mean error-run length 4.3 frames -- so a bit whose run
lands on a dark or soft scene has a majority of its frames voting wrong, and more frames
only make it more confidently wrong. Interleaving spreads each bit's frames across the
whole clip, so every bit sees the same mix of easy and hard content and no single scene
can own a bit.

They are the same family: with a fixed frame budget, `(i // 1) % bit_length` *is* the
interleaved map, so temp_redundancy is really a block-interleaver block size and
`interleave=True` is the temp_redundancy=1 end of it. Both are kept as an explicit flag
so an A/B is one argument rather than a reinterpretation of another one.
"""

from utils.bit import BIT_LENGTH


def bit_index_for_frame(frame_index, temp_redundancy, bit_length=BIT_LENGTH,
                        interleave=False, phase=0):
    """
    Payload bit carried by frame `frame_index`.

    `phase` shifts the map by whole frames. The embedder always writes at phase 0; a
    detector that has lost its temporal origin searches over it, which is what makes an
    interleaved payload self-synchronising (see blur_detect_mf.phase_search).
    """
    i = frame_index - phase
    if interleave:
        return i % bit_length
    return (i // temp_redundancy) % bit_length


def frames_per_bit(frame_count, temp_redundancy, bit_length=BIT_LENGTH,
                   interleave=False):
    """
    How many frames each bit gets, as a list indexed by bit position.

    Worth stating explicitly because the answer is the same for both maps at a fixed
    frame budget -- only the *arrangement* differs -- and that equality is what makes a
    contiguous/interleaved comparison at fixed frame count a fair one.
    """
    counts = [0] * bit_length
    for i in range(frame_count):
        counts[bit_index_for_frame(i, temp_redundancy, bit_length, interleave)] += 1
    return counts
