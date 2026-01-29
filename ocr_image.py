import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
import sys
import os
import argparse

# --- Configuration ---
MODEL_ID = "lightonai/LightOnOCR-2-1B" 

def load_model():
    print("[情報] モデルを読み込み中... (数分かかる場合があります)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Try float16 on GPU for memory savings, else float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto"
        )
    except Exception as e:
        print(f"[警告] auto読み込みに失敗しました。CPU/float32で試行します: {e}")
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to(device)

    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    print(f"[情報] モデル読み込み完了。デバイス: {device}")
    return model, processor, device, dtype

def run_ocr(model, processor, device, dtype, image_path):
    print(f"\n[処理中] 画像を読み込んでいます: {image_path}")
    
    try:
        if not os.path.exists(image_path):
            print(f"[エラー] ファイルが見つかりません: {image_path}")
            return

        image = Image.open(image_path).convert("RGB")

        print("[処理中] 推論を実行しています...")
        
        print("[処理中] 推論を実行しています...")
        
        # Use apply_chat_template to correctly handle image tokens
        # The processor expects a list of messages.
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": "Convert this receipt to markdown."}
                ]
            }
        ]

        # apply_chat_template returns input_ids and other necessary inputs already formatted
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != dtype:
             inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        # Output to console
        print("\n" + "="*20 + " OCR 結果 " + "="*20)
        print(output_text)
        print("="*50)

        # Save to file
        base_name = os.path.splitext(image_path)[0]
        md_filename = f"{base_name}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"[完了] 結果を保存しました: {md_filename}\n")

    except Exception as e:
        print(f"\n[エラー] 実行に失敗しました: {e}")

def main():
    parser = argparse.ArgumentParser(description="LightOnOCR Single Image Tool")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    model, processor, device, dtype = load_model()
    run_ocr(model, processor, device, dtype, args.image_path)

if __name__ == "__main__":
    main()
