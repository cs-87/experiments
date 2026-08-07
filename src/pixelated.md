Implement a Python module using OpenCV and NumPy that detects whether an image (or image region) has been intentionally pixelated.

### Background

I create pixelation by:

1. Downsampling an halves image region to a small resolution.
2. Upsampling it back to the original size using `cv2.INTER_NEAREST`.
3. check out pixelate_patch.py

This produces square blocks of constant color.

### Goal

Write a detector that returns a confidence score indicating whether the region was pixelated using this technique.

### Requirements

Implement:

```python
score = detect_pixelation(image)
```

where `score` is between 0 and 1.

The detector should combine multiple cues instead of relying on a single heuristic.

### Detection cues

Compute the following:

1. **Block variance**

   * Divide the image into fixed-size blocks (e.g. 8, 16, 32 pixels).
   * Compute variance within each block.
   * Pixelated images should have much lower average variance.

2. **Gradient concentration**

   * Compute Sobel gradients.
   * Measure whether gradients are concentrated on regularly spaced vertical and horizontal boundaries.
   * Produce a score.

3. **Laplacian energy**

   * Compute the Laplacian.
   * Pixelated images should contain low interior energy but strong responses near block edges.

4. **Frequency-domain analysis**

   * Compute a 2D FFT.
   * Look for periodic peaks corresponding to the block grid.
   * Convert this into a confidence score.

5. **Neighbor similarity**

   * Compute the percentage of neighboring pixels with identical or nearly identical values.
   * Pixelated regions should have much larger flat areas.

### Final score

Combine the normalized scores into a final confidence:

```python
confidence = (
    w1 * variance_score +
    w2 * gradient_score +
    w3 * laplacian_score +
    w4 * fft_score +
    w5 * neighbor_score
)
```

Normalize the output to [0,1].

### Additional requirements

* Use only Python, NumPy, and OpenCV.
* Avoid machine learning.
* The implementation should be modular with one function per metric.
* Include clear comments explaining why each metric works.
* Return intermediate metrics for debugging.
* Make block size configurable.
* The detector should work reasonably well after JPEG compression and mild H.264 compression.

### Deliverables

Return:

1. Complete Python implementation.
2. Explanation of each metric.
3. Example usage.
4. Visualization functions showing:

   * Block variance heatmap.
   * Gradient magnitude.
   * Laplacian.
   * FFT magnitude spectrum.
   * Final confidence.

The code should be production-quality and easy to tune.
