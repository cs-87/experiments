from utils.sift import SIFT
from numpy import ndarray
SQUARE_SIZE = 256


def get_patch(x, y, frame: ndarray, centre_point=False, square_size=SQUARE_SIZE):
    Y, X = frame.shape

    if centre_point:
        x = x - square_size // 2
        y = y - square_size // 2

        if x < 0 or y < 0:
            return None, None, None

    x1 = x + square_size
    y1 = y + square_size

    if x1 > X or y1 > Y:
        return None, None, None
    return frame[y:y1, x:x1], (y, y1), (x, x1)


def get_grid_patches(frame):

    height, width = frame.shape

    y, x = 0, 0

    for y in range(0, (height // SQUARE_SIZE)*SQUARE_SIZE, SQUARE_SIZE):
        for x in range(0, (width // SQUARE_SIZE)*SQUARE_SIZE, SQUARE_SIZE):
            yield get_patch(x, y, frame)


def get_sift_patches(frame):

    print("Calling SIFT to get keypoints for the frame...")
    sift = SIFT()
    kps, _ = sift.get_keypoints(frame)

    kps = sorted(
        kps,
        key=lambda kp: (kp.size, kp.response),
        reverse=True
    )

    kps = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]

    kps = non_overlapping_points(kps)

    for kp in kps:
        yield get_patch(kp[0], kp[1], frame, centre_point=True)


# kps = [(x,y),(x1,y1).....]
def non_overlapping_points(kps):
    ret_kps = []

    def overlaps(x1, y1, x2, y2):
        return (
            abs(x1 - x2) < SQUARE_SIZE and
            abs(y1 - y2) < SQUARE_SIZE
        )

    for x, y in kps:
        keep = True

        for x1, y1 in ret_kps:
            if overlaps(x, y, x1, y1):
                keep = False
                break

        if keep:
            ret_kps.append((x, y))

    return ret_kps


def select_candidate(half, pool, key=None):
    """
    Pick which patch of `half` to mark on this frame, given the spots already marked.

    Returns (patch, y, x, centre) or None if `half` is empty. `pool` is a list of
    (x, y) centres already used during the current bit run; the first candidate not
    suppressed by one of them wins. When every candidate is suppressed the pool is
    cleared and the walk restarts, so a long bit run keeps rotating instead of parking
    on half[0] -- a 128px block pixelated in one place for 100+ frames is exactly the
    artefact this rotation exists to avoid.

    key: optional scorer taking a patch and returning a float; candidates are then
    walked in descending score instead of SIFT's (size, response) order. SIFT ranks by
    blob scale, which says nothing about how well a given embedder's mark will survive
    that patch's content, so an embedder that knows what its own mark needs should pass
    a scorer measuring exactly that. Left None the order is unchanged.

    The choice depends only on this half's own history and on the original frame's
    pixels, never on the watermark bit -- a `key` that reads the bit, or that reads the
    already-marked frame, would break this. That is what lets detect() rebuild the
    identical sequence from the original video without knowing the bits it is trying to
    recover -- see detect() for why the per-run flush is load-bearing. Any change here
    must be mirrored on both sides.
    """
    if not half:
        return None

    order = half if key is None else sorted(half, key=lambda c: -key(c[0]))

    # Two passes at most: one to walk the list, one after a clear. A third could not
    # find anything the second did not.
    for _ in range(2):
        for patch, y, x in order:
            cx, cy = (x[0] + x[1]) // 2, (y[0] + y[1]) // 2
            if not any(abs(px - cx) < SQUARE_SIZE and abs(py - cy) < SQUARE_SIZE
                       for px, py in pool):
                return patch, y, x, (cx, cy)

            pool.clear()

    return None


def get_two_halves(frame):
    """
    Get two halves of the frame for watermark embedding.
    The first half is the left half of the frame, and the second half is a square patch from the right half.
    """
    height, width = frame.shape

    # First half: left half of the frame
    first_half = frame[:, :width // 2]

    # Second half: square patch from the right half of the frame
    second_half = frame[:, width // 2:width]

    return [(first_half, (0, height), (0, width // 2)), (second_half, (0, height), (width//2, width))]


def get_best_patch_in_two_halves(frame):
    """
    Get the best patch in the given half of the frame based on SIFT keypoints.
    """
    height, width = frame.shape

    sift = SIFT()
    kps, _ = sift.get_keypoints(frame)

    kps = sorted(
        kps,
        key=lambda kp: (kp.size, kp.response),
        reverse=True
    )

    kps = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]

    kps = non_overlapping_points(kps)

    first_half = []
    second_half = []

    for kp in kps:

        patch, y, x = get_patch(kp[0], kp[1], frame, centre_point=True)
        if patch is None:
            continue
        if x[1] <= width//2:
            first_half.append((patch, y, x))
        elif x[0] > width//2:
            second_half.append((patch, y, x))

    return first_half, second_half


def get_middle_patches(frame):
    H, W = frame.shape
    return (get_patch(W//2 - SQUARE_SIZE, H//2-SQUARE_SIZE//2, frame),), (get_patch(W//2, H//2-SQUARE_SIZE//2, frame),)


def get_nxn_block(patch, block_size):
    H, W = patch.shape

    for r in range(0, H-block_size+1, block_size):
        for c in range(0, W-block_size+1, block_size):

            yield get_patch(x=c, y=r, square_size=block_size, frame=patch)


def get_half_patches_grid(frame):
    H, W = frame.shape
    first_half = []
    second_half = []
    for patch, y, x in get_grid_patches(frame):
        if x[1] < W//2:
            first_half.append((patch, y, x))
        else:
            second_half.append((patch, y, x))

    return first_half, second_half
