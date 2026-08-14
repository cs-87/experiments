import subprocess
from pathlib import Path


def transcode_n_times(input_video, output_dir, n):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current = Path(input_video)

    for i in range(1, n + 1):
        output = output_dir / f"transcoded_{i}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(current),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output),
        ]

        print(f"Transcoding {i}/{n}")
        subprocess.run(cmd, check=True)

        current = output

    print(f"Final video: {current}")
