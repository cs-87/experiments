BIT_LENGTH = 32


def get_bit_string(output):
    bit_string = format(output, f"0{BIT_LENGTH}b")
    bit_string = bit_string
    return bit_string, BIT_LENGTH


def get_integer(bit_string):
    return int(bit_string[:BIT_LENGTH], 2)


def hex_to_uint32(value):
    return int(value, 16) & 0xFFFFFFFF


def uint32_to_hex(value):
    return f"{value & 0xFFFFFFFF:08X}"
