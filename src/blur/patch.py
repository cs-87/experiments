from utils.sift import SIFT
from numpy import ndarray
import numpy as np
SQUARE_SIZE = 240


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


def get_best_sift_patches_in_two_halves(frame):

    sift = SIFT()
    kps, _ = sift.get_keypoints(frame)

    kps = sorted(
        kps,
        key=lambda kp: (kp.size, kp.response),
        reverse=True
    )

    kps = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]

    kps = non_overlapping_points(kps)

    col = get_middle_split_col(width=frame.shape[1])

    best_first_half_patch = None
    best_second_half_patch = None

    for kp in kps:

        if best_first_half_patch is not None and best_second_half_patch is not None:
            break

        if kp[0] + SQUARE_SIZE < col and best_first_half_patch is None:
            best_first_half_patch = get_patch(
                kp[0], kp[1], frame, centre_point=True
            )

        if kp[0] > col and best_second_half_patch is None:
            best_second_half_patch = get_patch(
                kp[0], kp[1], frame, centre_point=True
            )

    return best_first_half_patch, best_second_half_patch


    



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

def get_best_patch_in_both_half(frame,score_func):
    first_half, second_half = get_half_patches_grid(frame)

    
    best_first_half_patch_details = None
    best_second_half_patch_details = None

    for i,half in enumerate((first_half,second_half)):

        best_score = -1
        best_patch = None

        for patch in half:
            orig_patch, y ,x =  patch


            score = score_func(orig_patch)

            if score > best_score:
                best_score = score
                best_patch = patch

        if i == 0:
            best_first_half_patch_details = best_patch
        else:
            best_second_half_patch_details = best_patch

    return best_first_half_patch_details, best_second_half_patch_details

    






def select_candidate(halves, pool, key=None):
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
    if not halves:
        return None

    order = halves if key is None else sorted(halves, key=lambda c: -key(c[0]))

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

    return [((first_half, (0, height), (0, width // 2)),), ((second_half, (0, height), (width//2, width)),)]


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


def get_middle_split_col(width, square_size=SQUARE_SIZE):
    """
    The grid line that separates the bit-1 cell from the bit-0 cell.

    Rounded to the *nearest* line rather than floored, so the marked pair straddles it
    as evenly as the grid allows. This is the boundary a detector has to test against,
    and it is not the same as width // 2: on 1920 with a 256 grid the centre pixel 960
    sits inside the right-hand cell, so splitting on 960 would put both candidate cells
    on its left and read every frame as a 1. Anything reading the bit back must call
    this rather than re-deriving a midpoint of its own.
    """
    return int(round(width / 2 / square_size)) * square_size


def get_middle_patches(frame, square_size=SQUARE_SIZE):
    H, W = frame.shape

    # Snap to the SQUARE_SIZE grid get_grid_patches tiles from (0, 0), so the embedded
    # block lands on exactly one grid cell instead of straddling four.
    col = int(round(W / 2 / square_size)) * square_size
    row = int(round(H / 2 / square_size)) * square_size

    return get_patch(col - square_size, row , frame), get_patch(col, row, frame)


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
        if x[1] <=get_middle_split_col(W):
            first_half.append((patch, y, x))
        else:
            second_half.append((patch, y, x))

    return first_half, second_half


# Cells per side carried by one bit. 1 reproduces get_middle_patches exactly, so the
# cluster is a strict generalisation of the two-cell embedder rather than a fork of it.
CLUSTER_K = 1


def _pair_offsets(max_ring=8):
    """
    (column_step, row_step) offsets ordered outward from the split column.

    column_step counts cells away from the split line, starting at 1; row_step counts
    cells away from the centre row, signed. Ordered by ring (column_step + |row_step|)
    so the innermost pairs come first, which is what puts the strongest, most
    centre-adjacent cells in the cluster and leaves the outer ones as controls.
    """
    offsets = []
    for ring in range(1, max_ring + 1):
        for dc in range(1, ring + 1):
            dr_mag = ring - dc
            for dr in ((0,) if dr_mag == 0 else (-dr_mag, dr_mag)):
                offsets.append((dc, dr))
    return offsets


def middle_pair_cells(frame, count, skip=0, square_size=SQUARE_SIZE):
    """
    `count` mirrored (left_cell, right_cell) grid-cell pairs straddling frame centre.

    Each element is ((patch, y, x), (patch, y, x)) -- the bit-1 candidate on the left of
    the split column and its mirror image on the right. Mirrored deliberately: the whole
    method is a difference between the two sides, so any asymmetry in how the two sets
    are chosen shows up as a bias that no amount of averaging removes.

    A pair is emitted only when both of its cells are wholly on frame. Dropping one side
    of a pair would reintroduce exactly that asymmetry.

    `skip` discards the first `skip` pairs, which is how the detector's control cells are
    taken from the same ordering as the marked cluster without overlapping it -- one
    generator for both means the two can never drift apart.
    """
    H, W = frame.shape
    col = get_middle_split_col(W, square_size)
    row = int(round(H / 2 / square_size)) * square_size

    shape = (square_size, square_size)
    pairs = []
    for dc, dr in _pair_offsets():
        y = row + dr * square_size
        lx = col - dc * square_size
        rx = col + (dc - 1) * square_size

        # get_patch only rejects an overhang past the far edge; a negative origin walks
        # off the near edge instead, where a numpy slice silently returns an empty array
        # rather than None. Checked here so an off-frame cell can never reach the DCT.
        if y < 0 or lx < 0 or rx < 0:
            continue

        left = get_patch(lx, y, frame, square_size=square_size)
        right = get_patch(rx, y, frame, square_size=square_size)

        if left[0] is None or right[0] is None:
            continue
        if left[0].shape != shape or right[0].shape != shape:
            continue

        pairs.append((left, right))
        if len(pairs) >= skip + count:
            break

    return pairs[skip:skip + count]


def get_middle_cluster(frame, k=CLUSTER_K, square_size=SQUARE_SIZE):
    """
    The K cells per side that carry one bit, as (left_cells, right_cells).

    k=1 returns exactly get_middle_patches' pair, wrapped in lists. Above that the extra
    cells are the next ones outward, which is where the redundancy has to come from:
    frames of one bit run are repeated measurements of a single scene and correlate
    almost perfectly, while cells in different parts of the frame sit on different
    content and their errors do not.
    """
    pairs = middle_pair_cells(frame, count=k, square_size=square_size)
    return [p[0] for p in pairs], [p[1] for p in pairs]
