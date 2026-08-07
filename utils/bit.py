BIT_LENGTH = 32


def get_bit_string(output):
    bit_string = format(output, f"0{BIT_LENGTH}b")
    return bit_string, BIT_LENGTH

def get_integer(bit_string, bit_length):
    return int(bit_string[:bit_length], 2)
