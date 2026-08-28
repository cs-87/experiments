import pathlib
import shutil
import subprocess

import cv2
from enum import Enum


# Rate control for the H.264 writer. Fixed CRF, not a fixed bitrate: a sweep that varies
# RADIUS or temporal redundancy changes how compressible the marked frames are, and a
# bitrate cap would then quietly hand a different quality to every cell of the sweep --
# the encode becomes a confound instead of a constant. 23 is x264's own default.
H264_CRF = 23
H264_PRESET = "medium"

class FrameType(Enum):
    YUV = 1
    RGB = 2

class Frame():
    def __init__(self, frame, frame_number=-1, frame_type = FrameType.YUV):
        self.frame = frame
        self.height, self.width, _ = frame.shape
        self.frame_type = frame_type

        self.frame_number = frame_number

        if self.frame_type == FrameType.YUV:
            self.y = frame[:, :, 0]
            self.u = frame[:, :, 1]
            self.v = frame[:, :, 2]

    def set_y(self, new_y):
        self.frame[:, :, 0] = new_y
        self.y = self.frame[:, :, 0]   # optional, keeps the view explicit


class FFmpegWriter():
    """
    H.264 writer piping raw BGR frames to ffmpeg at a fixed CRF.

    OpenCV's VideoWriter with 'mp4v' encodes MPEG-4 Part 2 at whatever internal default
    the build carries, which on 1080p lands anywhere between 3 and 20 Mbps depending on
    content. That is a fine way to store a video and a bad way to run an experiment: the
    codec is then an uncontrolled variable, and MPEG-4 Part 2 is not what any real
    distribution path uses anyway. This writes the same H.264 at the same CRF every time,
    so two runs of a sweep differ only by what the sweep changed.
    """

    def __init__(self, path, fps, width, height, crf=H264_CRF, preset=H264_PRESET):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; needed by FFmpegWriter")

        parent = pathlib.Path(path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)

        self.path = str(path)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{int(width)}x{int(height)}",
            # ffmpeg rejects a fractional fps of 0 or nan, which is what a container with
            # no frame-rate metadata hands back through CAP_PROP_FPS.
            "-r", f"{float(fps) if fps and fps > 0 else 30.0:.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            # 4:2:0 rather than ffmpeg's default 4:4:4 for rawvideo input: the mark lives
            # in luma, but a 4:4:4 file is not what a real pipeline would ever carry, and
            # chroma subsampling is part of the distortion the mark has to survive.
            "-pix_fmt", "yuv420p",
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr):
        self.proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        code = self.proc.wait()
        self.proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code} while writing {self.path}")


class Video_IO():
    def __init__(self, video_path, yuv=True, codec="h264", crf=H264_CRF):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if yuv:
            self.frame_type = FrameType.YUV
        else:
            self.frame_type = FrameType.RGB

        self.frame_number = -1

        # "h264" pipes to ffmpeg at a fixed CRF (see FFmpegWriter); "mp4v" keeps the old
        # cv2.VideoWriter path for callers that only want a file to look at.
        self.codec = codec
        self.crf = crf
        self.out = None


    def read_frame(self):

        ret, frame = self.cap.read()

        if self.frame_type == FrameType.YUV:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        if not ret:
            return None
        self.frame_number += 1
        return Frame(frame, frame_type=self.frame_type, frame_number=self.frame_number)

    def release(self):
        self.cap.release()
        # The writer is created lazily by write_frame, so it may not exist. Releasing it
        # here is what flushes the moov atom: without this the output mp4 stays
        # unreadable until the interpreter exits and GC finalises it, which breaks any
        # caller that writes a video and then reads it back in the same process.
        out = getattr(self, "out", None)
        if out is not None:
            out.release()
            self.out = None

    def write_frame(self, frame, output_path):
        # Keyed on the writer being absent rather than on frame_number == 0, so a caller
        # that starts writing partway through a video still gets a file.
        if self.out is None:
            if self.codec == "h264":
                self.out = FFmpegWriter(
                    output_path, self.fps, self.width, self.height, crf=self.crf)
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    output_path, fourcc, self.fps, (self.width, self.height))

        if self.frame_type == FrameType.YUV:
            frame_to_write = cv2.cvtColor(frame.frame, cv2.COLOR_YUV2BGR)
        else:
            frame_to_write = frame.frame

        self.out.write(frame_to_write)
