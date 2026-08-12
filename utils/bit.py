BIT_LENGTH = 32


def get_bit_string(output):
    bit_string = format(output, f"0{BIT_LENGTH}b")
    return bit_string, BIT_LENGTH

def get_integer(bit_string, bit_length):
    return int(bit_string[:bit_length], 2)

def hex_to_32bit(hex_value):
    int = int(hex_value, 16) & 0xFFFFFFFF
    return get_bit_string(int)

def int32_to_hex(x):
    return f"{x:08X}"


def get_bcr(original,recovered, expected_length, bit_lenght = BIT_LENGTH):
    bcr = 0
    for i in range(expected_length):
        if original[i] == recovered[i]:
            bcr=+1
    bcr=(bcr*100)/bit_lenght
    return bcr