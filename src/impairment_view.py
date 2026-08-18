import cv2
import numpy as np
import os
import tqdm


class Impairment_View:
    def __init__(self, src_path, imp_path, out_dir):
        self.src = cv2.VideoCapture(src_path)
        self.imp = cv2.VideoCapture(imp_path)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.frame_index = 0

    # override for custom frame reader
    def get_source_frame(self):
        ret, src_frame = self.src.read()
        # cvtColor must not run before the ret check: at EOF read() hands back None
        # and the conversion throws instead of ending the loop cleanly.
        if not ret:
            print("src ended")
            return None
        return cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

    def get_imp_frame(self):
        ret, imp_frame = self.imp.read()
        if not ret:
            print("imp ended")
            return None
        return cv2.cvtColor(imp_frame, cv2.COLOR_BGR2GRAY)

    def get_frame_diff(self):
        src_frame = self.get_source_frame()
        if src_frame is None:
            return None

        imp_frame = self.get_imp_frame()
        if imp_frame is None:
            return None

        imp_frame = cv2.resize(
            imp_frame, (src_frame.shape[1], src_frame.shape[0]))

        # Luma-only diff: comparing chroma channels here would weight
        # re-encode chroma noise over real visual difference.
        diff_gray = cv2.absdiff(src_frame[:, :, 0], imp_frame[:, :, 0])

        # Threshold
        _, diff_map = cv2.threshold(
            diff_gray,
            30,
            255,
            cv2.THRESH_BINARY
        )

        return diff_map

    def start(self):
        count = 0
        pbar = tqdm.tqdm(desc="Processing")
        try:
            while True:

                diff_map = self.get_frame_diff()

                if diff_map is None:
                    break

                # imwrite truncate-casts a float image to 8-bit without scaling,
                # so anything not already uint8 gets clipped explicitly here.
                '''if diff_map.dtype != np.uint8:
                    diff_map = np.clip(diff_map, 0, 255).astype(np.uint8)
                    '''

                # Zero-padded so the frames sort in playback order; frame_100
                # sorts before frame_11 otherwise.
                cv2.imwrite(f"{self.out_dir}/frame_{count:05d}.png", diff_map)
                count += 1
                self.frame_index = count
                pbar.update(1)
        finally:
            pbar.close()

    def release(self):
        self.src.release()
        self.imp.release()
