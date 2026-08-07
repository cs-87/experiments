SQUARE_SIZE = 256
from numpy import ndarray
from utils.sift import SIFT

def get_patch(x,y, frame:ndarray, centre_point=False, square_size=SQUARE_SIZE):
    Y,X = frame.shape

    if centre_point:
        x = x - square_size //2
        y = y - square_size //2

        if x < 0 or y < 0:
            return None, None, None
    
    x1 = x + square_size
    y1 = y + square_size

    if x1 > X or y1 > Y :
        return None, None, None

    return frame[y:y1,x:x1],(y,y1),(x,x1)

def get_grid_patches(frame):

    height, width = frame.shape

    y,x = 0,0

    for y in range(0,(height // SQUARE_SIZE)*SQUARE_SIZE ,SQUARE_SIZE):
        for x in range(0,(width // SQUARE_SIZE)*SQUARE_SIZE ,SQUARE_SIZE):
            yield get_patch(x,y,frame)


def get_sift_patches(frame):

    print("Calling SIFT to get keypoints for the frame...")
    sift = SIFT()
    kps,_ = sift.get_keypoints(frame)

    kps = sorted(
    kps,
    key=lambda kp: (kp.size, kp.response),
    reverse=True
    )   

    kps = [(int(kp.pt[0]),int(kp.pt[1])) for kp in kps]

    kps = non_overlapping_points(kps)

    for kp in kps:
        yield get_patch(kp[0],kp[1],frame, centre_point=True)


#kps = [(x,y),(x1,y1).....]
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

    return [(first_half,(0,height),(0,width // 2)), (second_half,(0,height),(width//2,width))]

def get_best_patch_in_two_halves(frame):
    """
    Get the best patch in the given half of the frame based on SIFT keypoints.
    """
    height, width = frame.shape

    sift = SIFT()
    kps,_ = sift.get_keypoints(frame)

    kps = sorted(
        kps,
        key=lambda kp: (kp.size, kp.response),
        reverse=True
    )   

    kps = [(int(kp.pt[0]),int(kp.pt[1])) for kp in kps]

    kps = non_overlapping_points(kps)

    first_half = None
    second_half = None

    for kp in kps:

        
        patch, y, x = get_patch(kp[0], kp[1], frame, centre_point=True)
        if patch is None:
            continue
        if x[1] <= width//2:
            first_half = (patch, y, x)
        elif x[0] > width//2:
            second_half = (patch, y, x)

        if first_half is not None and second_half is not None:
            return first_half,second_half

    return None, None, None