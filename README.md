## [WIP] Simple RNN learning phonetic regularities in Turkish
### Midterm results
The simple RNN with single hidden layer was learned to predict the next letter in the turkic word given the current letter. 
The number of neurons in the hidden layer was varying from 2 to 5. The hidden layer activation function is *htan*, the loss is computed via the negative log of the of the softmax output layer.

The network is easily learned to distinguish vovels and consonants using a hidden unit.

The network is able to distinguish the letter *j* which actually isn't native for turkish.

It is now unclear for me whether the network has learned the vovel or consonants harmony as it was described in the [paper](http://www.aclweb.org/anthology/W97-1012).

#### 2 units in hidden layer
##### Hidden unit 0 activation
![activation.png](plots\2_hidden_l\unit_0_e1000.png?raw=true "hidden unit 0 activation")
##### Hidden unit 1 activation
![activation.png](plots\2_hidden_l\unit_1_e1000.png?raw=true "hidden unit 1 activation")

#### 3 units in hidden layer
##### Hidden unit 0 activation
![activation.png](plots\3_hidden_l\unit_0_e1000.png?raw=true "hidden unit 0 activation")
##### Hidden unit 1 activation
![activation.png](plots\3_hidden_l\unit_1_e1000.png?raw=true "hidden unit 1 activation")
##### Hidden unit 2 activation
![activation.png](plots\3_hidden_l\unit_2_e1000.png?raw=true "hidden unit 2 activation")

#### 4 units in hidden layer
##### Hidden unit 0 activation
![activation.png](plots\4_hidden_l\unit_0_e1000.png?raw=true "hidden unit 0 activation")
##### Hidden unit 1 activation
![activation.png](plots\4_hidden_l\unit_1_e1000.png?raw=true "hidden unit 1 activation")
##### Hidden unit 2 activation
![activation.png](plots\4_hidden_l\unit_2_e1000.png?raw=true "hidden unit 2 activation")
##### Hidden unit 3 activation
![activation.png](plots\4_hidden_l\unit_3_e1000.png?raw=true "hidden unit 3 activation")

#### 5 units in hidden layer
##### Hidden unit 0 activation
![activation.png](plots\5_hidden_l\unit_0_e1000.png?raw=true "hidden unit 0 activation")
##### Hidden unit 1 activation
![activation.png](plots\5_hidden_l\unit_1_e1000.png?raw=true "hidden unit 1 activation")
##### Hidden unit 2 activation
![activation.png](plots\5_hidden_l\unit_2_e1000.png?raw=true "hidden unit 2 activation")
##### Hidden unit 3 activation
![activation.png](plots\5_hidden_l\unit_3_e1000.png?raw=true "hidden unit 3 activation")
##### Hidden unit 4 activation
![activation.png](plots\5_hidden_l\unit_4_e1000.png?raw=true "hidden unit 4 activation")

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
