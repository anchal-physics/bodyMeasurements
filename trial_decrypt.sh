#!/bin/sh

echo $TP
openssl enc -aes-256-cbc -pass "$TP" -d -A -in trial_secret_file.json.enc -out trial_secret_file_conv.json -iter 10000
echo "Decrypted file:"
cat trial_secret_file_conv.json
