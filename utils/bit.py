BIT_LENGTH = 40
SYNC = "11010011"


def get_bit_string(output):
    bit_string = format(output, f"0{BIT_LENGTH-len(SYNC)}b")
    bit_string = SYNC + bit_string
    return bit_string, BIT_LENGTH


def get_integer(bit_string):
    return int(bit_string[len(SYNC):BIT_LENGTH], 2)


def hex_to_uint32(value):
    return int(value, 16) & 0xFFFFFFFF


def uint32_to_hex(value):
    return f"{value & 0xFFFFFFFF:08X}"


def get_bcr(original, recovered, expected_length, bit_lenght=BIT_LENGTH):
    bcr = 0
    for i in range(expected_length):
        if original[i] == recovered[i]:
            bcr = +1
    bcr = (bcr*100)/bit_lenght
    return bcr
