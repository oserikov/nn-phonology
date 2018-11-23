APERTIUM_TUR_LEXC_BLOB_URL=https://raw.githubusercontent.com/apertium/apertium-tur/master/apertium-tur.tur.lexc
curl -s -XGET $APERTIUM_TUR_LEXC_BLOB_URL | grep -v '^!' | grep '[^<> ]\+:[^<> ]\+ \(N[^PU]\|V\)[^ ]\+ ;' | cut -f1 -d':'
