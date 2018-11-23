## usage
* clone and cd to the project 
```
$ git clone https://github.com/oserikov/nn-phonology.git
$ cd nn-phonology/
```
* made `.sh` scripts executable
```
$ find . -name '*.sh' | xargs chmod +x
```
* generate input data for NN. **NB!** `python3` is the command used in some `.sh` scripts to execute python scripts.
```
$ ./ml_input_data_generator.sh
```
## data folder
`data/tur_alphabet*.txt` files are are a turkish alphabet provided in wikipedia, an alphabet used in the paper and an alphabet used in apertium-tur (equals to the wiki one)

## .sh scripts
* `ml_input_data_generator.sh` generates pairs of onehot encoded phonemes for each pair of phonemes present in apertium-tur `.lexc` lexicon

### helper scripts
* `sh_scripts/lexemes_generator.sh` extracts all the lexemes provided in the apertium-tur lexicon.
* `sh_scripts/phonemes_sequences_extractor.sh` given the line-by-line list of lexemes extracts all the substrings of length 2 from each lexeme and line-by-line prints these substrings to stdout
* `sh_scripts/get_apertium_tur_alphabet.sh` extracts the alphabet used in the apertium-tur lexicon 
* `sh_scripts/turkish_onehot_encoder.sh` onehot encodes input strings using turkish alphabet
