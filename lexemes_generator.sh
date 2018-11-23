APERTIUM_TUR_LEXC_PATH=data/apertium-tur-master/apertium-tur.tur.lexc
cat $APERTIUM_TUR_LEXC_PATH| grep -v '^!' | grep '[^<> ]\+:[^<> ]\+ \(N[^PU]\|V\)[^ ]\+ ;' | cut -f1 -d':'
