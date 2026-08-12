#!/bin/bash
set -euo pipefail

input="${1:?usage: compress.sh <input> [output] [passes]}"
final="${2:-compressed_final.mp4}"
passes="${3:-100}"

a=".compress_a.mp4"
b=".compress_b.mp4"
trap 'rm -f "$a" "$b"' EXIT

src="$input"
for i in $(seq 1 "$passes"); do
    if [ $((i % 2)) -eq 1 ]; then dst="$a"; else dst="$b"; fi

    echo "pass $i/$passes"
    ffmpeg -y -v error -stats -i "$src" \
        -c:v libx264 -crf 28 -preset medium \
        -c:a aac -b:a 96k \
        "$dst"

    src="$dst"
done

mv "$src" "$final"
echo "wrote $final"
