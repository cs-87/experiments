from utils.video import Video_IO

import utils.dwt as dwt
import utils.dct as dct
from utils.bit import get_bit_string, BIT_LENGTH, get_integer
from utils.patch import get_grid_patches, get_sift_patches
from tqdm import tqdm
import numpy as np
from utils.scene_change import detect_scene_change

INPUT = "videos/4_sec_source.mp4"
OUTPUT = "./out/wmk.mp4" 

LEAKED = "leak.mp4"

SIFT_PATCHES = False 

ONES = (1 << 32) - 1
ZEROS = 0

TEMP_REDUNDANCY = 5
ALPHA = 20

coeff_1 = (4,5)
coeff_2 = (5,4)

def embed_patch(patch, bit):

    dwt_coeffs = dwt.get_dwt_coeff(patch)

    LL = dwt_coeffs.LL

    LL_dct_coeff = dct.dct2(LL)

    c1 = LL_dct_coeff[coeff_1] 
    c2 = LL_dct_coeff[coeff_2]

    if bit == 1 and c1 < c2:
        LL_dct_coeff[coeff_1], LL_dct_coeff[coeff_2] = c2, c1
    elif bit == 0 and c1 > c2:
        LL_dct_coeff[coeff_1], LL_dct_coeff[coeff_2] = c2, c1

    LL = dct.idct2(LL_dct_coeff)

    dwt_coeffs.LL = LL

    embed_patch = dwt.reconstruct_frame(dwt_coeffs)

    return embed_patch


def embed(video_path, output_path, watermark:int):

    bit_string, length = get_bit_string(watermark)
    video_io = Video_IO(video_path)
    frame_count = video_io.frame_count
    for i in tqdm(range(frame_count),total=frame_count, unit="frame", desc="embedding"):
        frame = video_io.read_frame()
        if frame is None:
            break

        bit_index = (i // TEMP_REDUNDANCY) % length

        bit = int(bit_string[bit_index])

        wmk_frame_y = frame.y.copy()

        for patch, Y, X in (
            get_sift_patches(frame.y) if SIFT_PATCHES else get_grid_patches(frame.y)
        ):

            if patch is None:
                continue

            dwt_coeffs = dwt.get_dwt_coeff(patch)

            LL = dwt_coeffs.LL

            LL_dct_coeff = dct.dct2(LL)

            c1 = LL_dct_coeff[coeff_1] 
            c2 = LL_dct_coeff[coeff_2]


            diff = c1 - c2

            if bit == 1:
                if diff < ALPHA:
                    factor = (ALPHA - diff) / 2
                    c1 += factor
                    c2 -= factor

            elif bit == 0:
                if diff > -ALPHA:
                    factor = (diff + ALPHA) / 2
                    c1 -= factor
                    c2 += factor

            LL_dct_coeff[coeff_1] = c1
            LL_dct_coeff[coeff_2] = c2

            LL = dct.idct2(LL_dct_coeff)

            dwt_coeffs.LL = LL

            recon = dwt.reconstruct_frame(dwt_coeffs)
            wmk_frame_y[Y[0]:Y[1], X[0]:X[1]] = np.clip(recon, 0, 255)

        prv_frame = frame
        frame.set_y(wmk_frame_y)

        video_io.write_frame(frame, output_path)
    video_io.release()



def detect(org_video_path, watermark_path, bit_length=BIT_LENGTH):
    input_io = Video_IO(org_video_path)
    frame_count = input_io.frame_count

    wmk_io = Video_IO(watermark_path)

    print(wmk_io.frame_count, input_io.frame_count)
    temp_redundancy = TEMP_REDUNDANCY * (wmk_io.frame_count // frame_count)

    if temp_redundancy == 0:
        temp_redundancy = 1

    bit_array=[[] for _ in range(bit_length)]
    bit_index = 0
    for i in tqdm(range(wmk_io.frame_count), desc=f"detecting:{watermark_path}"):
        
        frame = wmk_io.read_frame()
        if frame is None:
            break

        bit_index = (i // temp_redundancy) % bit_length

        
        for patch, Y, X in (
            get_sift_patches(frame.y) if SIFT_PATCHES else get_grid_patches(frame.y)
        ):

            if patch is None:
                continue

            dwt_coeffs = dwt.get_dwt_coeff(patch)

            LL = dwt_coeffs.LL

            LL_dct_coeff = dct.dct2(LL)

            c1 = LL_dct_coeff[coeff_1] 
            c2 = LL_dct_coeff[coeff_2]

            if c1 > c2:
                bit_array[bit_index].append(1)
            elif c1 < c2:
                bit_array[bit_index].append(0)
            else:
                bit_array[bit_index].append(None)

    #print(bit_array)

    bit_str = ""

    for bit_index in bit_array:
        diff = bit_index.count(1) - bit_index.count(0)

        if diff > 0:
            bit_str+= "1"
        elif diff < 0:
            bit_str+="0"
        else:
            bit_str+="?"

    if frame_count < TEMP_REDUNDANCY*(bit_length):
        expected_length = (frame_count//TEMP_REDUNDANCY)
    else:
        expected_length = bit_length

    watermark =  get_integer(bit_str, expected_length)
    print(bit_str, watermark, expected_length)
    return watermark


embed(video_path=INPUT, output_path=OUTPUT, watermark=8710)
#detect(org_video_path=INPUT, watermark_path=LEAKED)