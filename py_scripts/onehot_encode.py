import sys

alphabet_file = sys.argv[1]
with open(alphabet_file, encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]

for line in sys.stdin:
    onehot_encoding_line = []
    for char in line.strip():
        onehot_encoding_char = ["0"] * len(alphabet)
        onehot_encoding_char[alphabet.index(char)] = "1"
        onehot_encoding_line += ["".join(onehot_encoding_char)]
    print(" ".join(onehot_encoding_line))
