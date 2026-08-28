import cv2
from skimage.metrics import structural_similarity as ssim

# Factor the frame is downsampled by before SSIM. Full-resolution SSIM on a 1080p pair
# costs ~380ms, which on a feature-length source is hours of wall clock spent deciding a
# boolean -- more than the embedder it is advising. At 4 it costs ~11ms and the decision
# does not change, because the separation it is thresholding is enormous: measured on
# 4_sec_source, consecutive frames read 0.883 at full resolution and 0.976 at scale 4,
# while synthetic cuts (flip, invert, large translation, noise) read 0.19/-0.29/0.23/0.01
# and 0.13/-0.46/0.15/0.08. Downsampling pushes the two populations further apart, not
# closer: the blur removes the per-pixel differences that drag a same-scene pair's score
# down while leaving the structural change a real cut consists of untouched.
SCALE = 4


def detect_scene_change(frame1, frame2, threshold=0.6, scale=SCALE):
    """
    True when the two Y planes are structurally unrelated, i.e. a cut sits between them.

    Callers pass luma planes, not Frame objects, and the two must be the same shape.
    """
    if scale > 1:
        h, w = frame1.shape[:2]
        size = (w // scale, h // scale)
        # INTER_AREA: this is a decimation, and averaging over the discarded pixels is
        # what makes the result a low-pass view rather than a sparse sample of one. A
        # nearest-neighbour subsample would alias fine texture into the structural term
        # SSIM is measuring and make the same-scene score noisy.
        frame1 = cv2.resize(frame1, size, interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, size, interpolation=cv2.INTER_AREA)

    return ssim(frame1, frame2) < threshold
