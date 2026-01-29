# Jetson Nano Setup Guide for LightOnOCR (Transformers版)

このガイドは、Jetson Nano上で LightOnOCR-2-1B を **Transformersライブラリ** を使用して動作させるための手順書です。

## ⚠️ メモリに関する重要事項

Jetson Nano (4GB) でTransformersを使用するため、メモリ不足になる可能性が高いです。**Swap領域の拡張** を強く推奨します。

## 1. 必要なツールのインストール (システム側)

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3-pip libopencv-dev python3-opencv libopenblas-dev
```

## 2. 一括インストール (uv使用)

`pyproject.toml` に依存関係が定義されています。以下のスクリプトで一括インストールできます。

```bash
bash install.sh
```

または手動で行う場合:

```bash
uv sync --prerelease=allow
```

## 3. 実行

インストールが完了したら、以下のコマンドで実行してください。

```bash
source .venv/bin/activate
python3 receipt_ocr.py
```

- 初回はモデルのダウンロードに時間がかかります。
- メモリ不足で落ちる場合は、他のアプリを終了してください。
