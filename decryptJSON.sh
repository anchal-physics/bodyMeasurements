#!/bin/sh

openssl enc -aes-256-cbc -pass "$PASSPHRASE" -d -A -in bodymeasurements-d291f81a84a8.json.enc -out bodymeasurements-d291f81a84a8.json -iter 10000
