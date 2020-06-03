<?php

if (count($argv) != 4) {
  die("Usage: $argv[0] output_dir length count\n");
}

require_once dirname(__FILE__) . '/securimage/securimage.php';
require_once dirname(__FILE__) . '/securimage/WavFile.php';

$img = new Securimage();

$img->charset = "0123456789";
$img->code_length = intval($argv[2]);

// Other audio settings
//$img->audio_use_sox   = true;
$img->audio_use_noise = true;
$img->degrade_audio   = true;
//$img->sox_binary_path = 'sox';
//Securimage::$lame_binary_path = '/usr/bin/lame'; // for mp3 audio support

// To use an alternate language, uncomment the following and download the files from phpcaptcha.org
// $img->audio_path = $img->securimage_path . '/audio/es/';

$r = new ReflectionMethod('Securimage', 'getAudibleCode');
$r->setAccessible(true);

$iters = intval($argv[3]);

for($i = 1; $i <= $iters; $i++) {
  $img->createCode();
  $audio = $r->invoke($img);
  $code = $img->getCode(true)['display'];
  $file = $code . '_' . bin2hex(random_bytes(4)) . '.wav';
  $path = $argv[1] . '/' . $file;
  echo $file . "\n";
  file_put_contents($path, $audio);
}
