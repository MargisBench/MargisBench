import os
import argparse
import sys
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify

# --- Configuration ---
DATASET_DIR = './test'
CLASSES = ['def_front', 'ok_front']
RESIZE_DIM = 256

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(pil_img, input_size, input_details):
    """
    1. Resizes image (Bilinear to match Torchvision).
    2. Normalizes (0-255 -> 0.0-1.0 -> minus Mean / Std).
    3. Quantizes manually if the model input is Int8.
    """
    target_h, target_w = input_size[1], input_size[0] # PIL uses (W, H)
    
    # 1. Resize to 256 (Standard Torchvision behavior)
    # We resize the smaller edge to 256, maintaining aspect ratio
    resize_dim = RESIZE_DIM
    w, h = pil_img.size
    if w < h:
        new_w = resize_dim
        new_h = int(h * (resize_dim / w))
    else:
        new_h = resize_dim
        new_w = int(w * (resize_dim / h))

    # 1. Resize matching PyTorch default (Bilinear)
    # Note: Image.ANTIALIAS is deprecated in newer PIL versions
    img = pil_img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - target_w) / 2
    top = (new_h - target_h) / 2
    right = (new_w + target_w) / 2
    bottom = (new_h + target_h) / 2
    img = img.crop((left, top, right, bottom))
    
    # 2. Convert to Float32 and Normalize
    # Shape: (H, W, 3)
    data = np.array(img).astype(np.float32) / 255.0 
    data = (data - MEAN) / STD

    # 3. Handle Quantization if model expects Int8/UInt8
    # If your model takes Float32, this block is skipped automatically.
    if input_details['dtype'] == np.int8 or input_details['dtype'] == np.uint8:
        scale, zero_point = input_details['quantization']
        # Quantize: q = (real / scale) + zp
        data = (data / scale) + zero_point
        
        # Clip to valid range and cast
        if input_details['dtype'] == np.int8:
            data = np.clip(data, -128, 127)
        else:
            data = np.clip(data, 0, 255)
            
        data = data.astype(input_details['dtype'])

    return data

def get_accuracy(model_path):
    #print(f"Loading model: {model_path}")
    
    try:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    input_details = interpreter.get_input_details()[0]
    #print(f"input dtype is {input_details['dtype']}")
    h, w = input_details['shape'][1], input_details['shape'][2]
    size = (w, h)
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    results = {"total": 0, "correct": 0}
    
    print("[\x1b[34mINFO\x1b[0m] Processing images...", end='', flush=True)
    
    for idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(folder): continue
        
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            img_path = os.path.join(folder, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                input_data = preprocess_image(img, size, input_details)
                interpreter.set_tensor(input_details['index'], [input_data])
                interpreter.invoke()
                
                objs = classify.get_classes(interpreter, top_k=1)
                pred_idx = objs[0].id

                if objs and (0 <= pred_idx < num_classes):
                    cm[idx][pred_idx] += 1

                    if pred_idx == idx:
                        results["correct"] += 1
                results["total"] += 1
            except Exception as e:
                print(f"\n[WARN] Failed to process {fname}: {e}")

    print(" Done.")

    # --- PRINT CONFUSION MATRIX ---
    print()
    print("================== CONFUSION MATRIX ==================")
    matrix_label = "True \\ Pred"
    print(f"{matrix_label:<15}", end="")
    for c in CLASSES:
        print(f"{c[:10]:<12}", end="") # Truncate long names for display
    print()
    
    for i, row_name in enumerate(CLASSES):
        print(f"{row_name:<15}", end="")
        for val in cm[i]:
            print(f"{val:<12}", end="")
        print()
    print("======================================================")
    
    
    acc = 0
    if results["total"] > 0:
        acc = (results["correct"] / results["total"]) * 100
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .tflite model')
    parser.add_argument('--resize-dim', type=int, required=True, help='Resize size of images for preprocessing phase')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='List of class names')
    args = parser.parse_args()

    CLASSES = args.classes
    #print(CLASSES)

    RESIZE_DIM = args.resize_dim
    #print(RESIZE_DIM)

    if len(CLASSES) == 1 and ' ' in CLASSES[0]:
        CLASSES = CLASSES[0].split(' ')

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)

    final_acc = get_accuracy(args.model)
    
    # We print a specific tag so Bash can grep it easily if needed, 
    # but also human readable
    print(f"[\x1b[34mINFO\x1b[0m] ACCURACY_RESULT: {final_acc:.2f}%")
