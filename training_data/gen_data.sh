#!/bin/bash

mkdir -p ../data/securimage_digits/train
mkdir -p ../data/securimage_digits/test
sed -i 's/0123456789abcdefghijklmnopqrstuvwxyz/0123456789/g' gen_data_securimage.php
php gen_data_securimage.php ../data/securimage_digits/train 4 50
php gen_data_securimage.php ../data/securimage_digits/test 4 10

mkdir -p ../data/securimage_digits_distorted/train
mkdir -p ../data/securimage_digits_distorted/test
sed -i 's/mt_rand(95/mt_rand(80/g' securimage/securimage.php
php gen_data_securimage.php ../data/securimage_digits_distorted/train 4 50
php gen_data_securimage.php ../data/securimage_digits_distorted/test 4 10

mkdir -p ../data/securimage_all/train
mkdir -p ../data/securimage_all/test
sed -i 's/mt_rand(80/mt_rand(95/g' securimage/securimage.php
sed -i 's/0123456789/0123456789abcdefghijklmnopqrstuvwxyz/g' gen_data_securimage.php
php gen_data_securimage.php ../data/securimage_all/train 4 50
php gen_data_securimage.php ../data/securimage_all/test 4 10
