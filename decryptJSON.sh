#!/bin/sh

openssl enc -nosalt -aes-256-cbc -d -md sha256 -in bodymeasurements-d291f81a84a8.json.enc -out bodymeasurements-d291f81a84a8.json -base64 -K "$ENCKEY" -iv "$INIT_VECT"
