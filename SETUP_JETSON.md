# Jetson Nano Setup Guide for LightOnOCR

このガイドは、Jetson Nano上で LightOnOCR-2-1B を動作させるための手順書です。

## 1. 必要なツールのインストール

ターミナルを開き、以下のコマンドを実行してください。

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3-pip libopencv-dev python3-opencv
```

## 2. llama.cpp のビルド

`llama.cpp` をクローンしてビルドします。Jetson Nano (CUDA 10.2) 向けの設定です。

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CUDAサポート付きでビルド (失敗する場合は GGML_CUDA=0 にしてください)
make clean
GGML_CUDA=1 make -j$(nproc)
```

ビルドが完了すると、`llama-cli` (または `main`) という実行ファイルが生成されます。

## 3. モデルのダウンロード

`models` ディレクトリを作成し、そこにモデルをダウンロードします。

```bash
mkdir -p models
cd models

# Language Model (Q8_0)
wget -O LightOnOCR-2-1B-Q8_0.gguf https://huggingface.co/noctrex/LightOnOCR-2-1B-GGUF/resolve/main/LightOnOCR-2-1B-Q8_0.gguf

# Vision Projector (mmproj)
wget -O mmproj-model-f16.gguf https://huggingface.co/noctrex/LightOnOCR-2-1B-GGUF/resolve/main/mmproj-model-f16.gguf

cd ..
```

## 4. 実行スクリプトの配置

作成した `receipt_ocr.py` を `llama.cpp` ディレクトリと同じ階層（またはわかりやすい場所）に配置します。

## 5. 実行

カメラを接続し、以下のコマンドで実行します。

```bash
python3 receipt_ocr.py
```

- カメラプレビューが表示されます。
- レシートを映して **「Enter」キー** を押すと撮影＆認識が始まります。
- **「q」キー** で終了します。
