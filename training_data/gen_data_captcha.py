#!/usr/bin/env python3

import os
import sys

if 'pypy' in sys.executable:
    sys.path.append("./captcha")
    from captcha.audio import AudioCaptcha
else:
    from captcha.captcha.audio import AudioCaptcha

import random
import secrets

CHARSET = "0123456789"

def main():
    if len(sys.argv) != 4:
        print("Usage: {} output_dir length count".format(sys.argv[0]))
        return

    output_dir = sys.argv[1]
    length = int(sys.argv[2])
    count = int(sys.argv[3])

    if not os.path.exists(output_dir):
        print("{} does not exist".format(output_dir))

    audio = AudioCaptcha(voicedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "voices_en"))

    for i in range(count):
        inp = ''.join(random.choices(CHARSET, k=length))
        filename = "{}_{}.wav".format(inp, secrets.token_hex(4))
        audio.write(inp, os.path.join(output_dir, filename))
        print("Generated: {}".format(filename))

if __name__ == "__main__":
    main()
