# Jetson Nano Setup Guide for LightOnOCR (Transformers版)

このガイドは、Jetson Nano上で LightOnOCR-2-1B を **Transformersライブラリ** を使用して動作させるための手順書です。

## ⚠️ メモリに関する重要事項

Jetson Nano (4GB) でTransformersを使用するため、メモリ不足になる可能性が高いです。**Swap領域の拡張** を強く推奨します。

## 1. 必要なツールのインストール

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3-pip libopencv-dev python3-opencv libopenblas-dev
```

## 2. Pythonライブラリのインストール

`uv` (高速なインストーラ) を推奨していますが、通常の `pip` でも可能です。

```bash
# uvのインストール (オプション)
pip install uv

# Transformers と依存関係のインストール
# LightOnOCR-2 はtransformersの最新版(dev/source) が必要
pip install git+https://github.com/huggingface/transformers
pip install torch pillow pypdfium2 accelerate protobuf scipy
```

※ `torch` はJetson Nano用のものをインストール済みであることを想定しています。未インストールの場合は NVIDIA のフォーラム等から JetPack 4.6 (Python 3.6) に対応した `pip install` 可能な wheel を探して入れてください（例: `torch-1.10.0` など）。
※ 本スクリプトは Python 3.6 以降で動作します。

## 3. モデルについて

スクリプト初回実行時に、Hugging Face Hub からモデル `lightonai/LightOnOCR-2-1B` が自動的にダウンロードされます（約数GB）。
インターネット接続が必要です。

## 4. 実行

```bash
python3 receipt_ocr.py
```

- 初回はモデルのダウンロードに時間がかかります。
- メモリ不足で落ちる場合は、他のアプリを終了してください。
