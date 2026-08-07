from skimage.metrics import structural_similarity as ssim

def detect_scene_change(frame1, frame2, threshold=0.6):

    # Compute SSIM between the two frames
    score, _ = ssim(frame1, frame2, full=True)

    # If the SSIM score is below the threshold, we consider it a scene change
    return score < threshold