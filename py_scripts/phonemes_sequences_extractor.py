# encoding: utf-8

import sys

phoneme_sequence_basic_length = 2 if len(sys.argv) < 2 else int(sys.argv[1])


def get_all_substrings(input_string, length):
    return [input_string[i:i + length] for i in range(len(input_string) - length + 1)]


def get_all_seqs_of_phonemes(lexemes, length_of_phonemes_seq):
    seqs = []
    for lexeme in lexemes:
        seqs += get_all_substrings(lexeme, length_of_phonemes_seq)
        seqs.append("")
    return seqs


lexemes_list = [l.strip() for l in sys.stdin]

for seq_of_phonemes in get_all_seqs_of_phonemes(lexemes_list, phoneme_sequence_basic_length):
    print(seq_of_phonemes)
