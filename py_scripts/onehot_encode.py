import sys

alphabet_file = sys.argv[1]


def get_alphabet(filename):
    global alphabet
    with open(filename, encoding='utf-8') as f:
        alphabet = [l.strip() for l in f]
    return alphabet

alphabet = get_alphabet(alphabet_file)


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


for line in sys.stdin:
    onehot_encoding_line = []
    for char in line.strip():
        onehot_encoding_char = onehot_encode_char(alphabet, char)
        onehot_encoding_line += ["".join(onehot_encoding_char)]
    print(" ".join(onehot_encoding_line))
