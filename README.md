#### possible bug
there are cyrillic *а* and *е* letters in lexemes listed in turkish lexicon in apertium-tur:
```bash
$ diff data/tur_alphabet_wiki.txt <(./get_apertium_tur_alphabet.sh)
29a30,31
> а
> е
```

## data folder
`data/tur_alphabet*.txt` files are are a turkish alphabet provided in wikipedia and an alphabet used in the paper

## .sh scripts

* `lexemes_generator.sh` extracts all the lexemes provided in the apertium-tur lexicon.
  * **NB! `APERTIUM_TUR_LEXC_PATH`** variable should be configured properly
* `phonemes_sequences_generator.sh` given the line-by-line list of lexemes extracts all the substrings of length 2 from each lexeme and line-by-line prints these substrings to stdout
* `get_apertium_tur_alphabet.sh` extracts the alphabet used in the apertium-tur lexicon 
