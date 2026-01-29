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
        print(f"Error: llama-cli not found at {LLAMA_CLI_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
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

    print("\n[INFO] Running OCR... Please wait.")
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
            print("\n" + "="*20 + " OCR RESULT " + "="*20)
            print(result.stdout)
            print("="*52 + "\n")
        else:
            print(f"\n[ERROR] OCR failed with code {result.returncode}")
            print(result.stderr)
            
    except Exception as e:
        print(f"\n[ERROR] execution failed: {e}")

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Camera initialized.")
    print("Press 'Enter' to capture and run OCR.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
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
            print(f"\n[INFO] Captured {filename}")
            
            # Run OCR
            run_ocr(filename)
            
            # Optional: Clean up file
            # os.remove(filename) 

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
