# sradcon
This is pure free hyper-quality resampler, re-quantizer, and having high-freq predictor effect.
JP日本語:

このプロジェクトは、Saracon並の音質のリサンプラーをフリーで置き換えることを目指したプロジェクトです。
このリサンプラー、再量子化器は以下のアルゴリズムで成り立っています。
リサンプラー部：
オーバーラップされたチャンクに分割し、線形位相と最小位相を細かいバンド事、主にメル尺度の128バンドごとに使い分け、全体的なノイズトーナリティ、ゲイン、バンド情報を元にリサンプリング後の差分を最小化するような選択を全て試します。それらから最善と判断された方法がそれぞれのバンドごとに使われます。
再量子化器部：
オーバーラップされたチャンクに分割し、周波数分布を解析し、ノイズが聞こえないところにノイズを盛るようなシェーピング係数を設計し、それらを元にノイズシェーピングを行います。
HFE（高音部再構築）機能はいまだ作成中です。

※このプロジェクトはLLMを活用しています。

English:

This project aims to replace Saracon-level audio quality resampler with a free alternative.
This resampler and re-quantizer consists of the following algorithms:
Resampler Section:
The audio is divided into overlapped chunks, and linear phase and minimum phase filters are selectively used for fine frequency bands, primarily 128 bands on the mel scale. All possible choices are tested to minimize the difference after resampling based on overall noise tonality, gain, and band information. The method judged to be optimal is used for each respective band.
Re-quantizer Section:
The audio is divided into overlapped chunks, frequency distribution is analyzed, and shaping coefficients are designed to place noise in areas where it cannot be heard, then noise shaping is performed based on these coefficients.
HFE (High Frequency Enhancement) function is still under development.

※This project utilizes LLM technology.
