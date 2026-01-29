import cv2
import subprocess
import os
import sys
import datetime

# --- Configuration ---
# Path to llama.cpp executable (llama-cli or main)
# Adjust this path to where your compiled llama-cli is located
LLAMA_CLI_PATH = "./llama.cpp/llama-cli" 

# Path to the GGUF models
MODEL_PATH = "./llama.cpp/models/LightOnOCR-2-1B-Q8_0.gguf"
MMPROJ_PATH = "./llama.cpp/models/mmproj-model-f16.gguf"

# Camera Configuration
CAMERA_ID = 0  # 0 is usually the default USB camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def run_ocr(image_path):
    """
    Runs llama.cpp CLI to perform OCR on the given image.
    """
    if not os.path.exists(LLAMA_CLI_PATH):
        print(f"エラー: llama-cli が見つかりません: {LLAMA_CLI_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルが見つかりません: {MODEL_PATH}")
        return

    # Construct the command
    # LightOnOCR prompt template usually involves just the image processing or a specific prompt.
    # We will use a generic prompt for document conversion if needed, but often VLMs just describe the image if no prompt is given.
    # For LightOnOCR, the docs suggest specific conversation templates, but for raw llama-cli, we can try a simple user prompt.
    prompt = "Convert this receipt to markdown."
    
    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--image", image_path,
        "-p", prompt,
        "--n-predict", "1024", # Max tokens to generate
        "--temp", "0.1",      # Low temperature for deterministic output
        "-c", "2048"          # Context window
    ]

    print("\n[情報] OCRを実行中... お待ちください。")
    try:
        # Run command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print("\n" + "="*20 + " OCR 結果 " + "="*20)
            print(result.stdout)
            print("="*50 + "\n")
        else:
            print(f"\n[エラー] OCRがコード {result.returncode} で失敗しました")
            print(result.stderr)
            
    except Exception as e:
        print(f"\n[エラー] 実行に失敗しました: {e}")

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("カメラを初期化しました。")
    print("「Enter」キーを押して撮影＆OCR実行")
    print("「q」キーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームの取得に失敗しました。")
            break

        # Display the resulting frame
        cv2.imshow('Receipt OCR - Press Enter to Capture', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 13: # Enter key
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"receipt_{timestamp}.jpg"
            
            # Save the frame
            cv2.imwrite(filename, frame)
            print(f"\n[情報] 画像を保存しました: {filename}")
            
            # Run OCR
            run_ocr(filename)
            
            # Optional: Clean up file
            # os.remove(filename) 

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
