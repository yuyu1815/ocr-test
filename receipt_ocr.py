import cv2
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
import datetime
import os
import sys

# --- Configuration ---
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Model ID
# Using the base model. To use 8-bit, bitsandbytes requires specific setup on Jetson.
# If Q8_0 GGUF was intended, this script CANNOT load it directly. 
# We assume 'bitsandbytes' is available or we load in float16/float32.
MODEL_ID = "lightonai/LightOnOCR-2-1B" 

def load_model():
    print("[情報] モデルを読み込み中... (時間がかかります)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Jetson Nano usually supports float16, but bfloat16 might be an issue.
    # Defaulting to float16 for memory saving if on MPS/CUDA.
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # Attempt to load with 8bit quantization if bitsandbytes is available
        # Note: 'load_in_8bit=True' requires bitsandbytes library
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            # load_in_8bit=True, # Uncomment if bitsandbytes is successfully installed
            device_map="auto"
        )
    except Exception as e:
        print(f"[警告] 8bit/auto読み込みに失敗しました。CPU/標準モードで試行します: {e}")
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to(device)

    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    print(f"[情報] モデル読み込み完了。デバイス: {device}")
    return model, processor, device, dtype

def run_ocr(model, processor, device, dtype, image_path):
    print(f"\n[処理中] 画像を読み込んでいます: {image_path}")
    print("[処理中] OCR推論を実行しています... (数秒〜数分かかります)")
    
    try:
        # Load image via PIL
        image = Image.open(image_path).convert("RGB") # Ensure RGB

        # Using generic processor call
        inputs = processor(
            images=image,
            text="<|user|>\n<|image|>\nConvert this receipt to markdown.<|end|>\n<|assistant|>\n",
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != dtype:
             inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        
        # Decode
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        # Save to Markdown
        base_name = os.path.splitext(image_path)[0]
        md_filename = f"{base_name}.md"
        
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(output_text)

        print("\n" + "="*20 + " OCR 結果 " + "="*20)
        print(output_text)
        print("="*50)
        print(f"[完了] 結果を保存しました: {md_filename}\n")

    except Exception as e:
        print(f"\n[エラー] OCR実行に失敗しました: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Load model once at startup
    try:
        model, processor, device, dtype = load_model()
    except Exception as e:
        print(f"致命的なエラー: モデルの初期化に失敗しました。\n{e}")
        return

    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("カメラを初期化しました。")
    print("「Enter」キーを押して撮影 ＆ OCR開始")
    print("「q」キーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームの取得に失敗しました。")
            break

        cv2.imshow('Receipt OCR (Transformers) - Press Enter to Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 13: # Enter
            print("\n" + "-"*30)
            print("[操作] 撮影しました！処理を開始します...")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"receipt_{timestamp}.jpg"
            
            # Save the frame first
            cv2.imwrite(filename, frame)
            print(f"[保存] 画像を保存しました: {filename}")
            
            # Run OCR on the saved file
            run_ocr(model, processor, device, dtype, filename)
            
            print("[待機] 次の撮影準備完了。Enterキーを押してください。")
            print("-"*30 + "\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
