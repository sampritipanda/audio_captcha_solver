#!/bin/bash

mkdir -p ../data/securimage_digits/train
mkdir -p ../data/securimage_digits/test
php gen_data_securimage.php ../data/securimage_digits/train 4 1000
php gen_data_securimage.php ../data/securimage_digits/test 4 500

mkdir -p ../data/securimage_digits_distorted/train
mkdir -p ../data/securimage_digits_distorted/test
php gen_data_securimage.php ../data/securimage_digits_distorted/train 4 1000
php gen_data_securimage.php ../data/securimage_digits_distorted/test 4 500

mkdir -p ../data/securimage_all/train
mkdir -p ../data/securimage_all/test
php gen_data_securimage.php ../data/securimage_all/train 4 1000
php gen_data_securimage.php ../data/securimage_all/test 4 500
