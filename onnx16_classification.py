"""import onnx
from onnxconverter_common import float16

model = onnx.load("models/NasNetLarge/model.onnx")

model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=False
)

onnx.save(model_fp16, "models/NasNetLarge/model16_2.onnx")"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_onnx_prediction(input_dir='input1', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idx = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}

    images, names, y_true = [], [], []

    for class_name, class_idx in class_to_idx.items():
        for fname in os.listdir(os.path.join(input_dir, class_name)):
            img = Image.open(os.path.join(input_dir, class_name, fname)).convert("RGB").resize((224, 224))
            images.append(np.array(img, dtype=np.float32) / 255.0)
            names.append(fname)
            y_true.append(class_idx)

    X = np.stack(images).astype(np.float16)

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True

    session = ort.InferenceSession(
        "models/NasNetLarge/model16_2.onnx",
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name

    y_pred = session.run(None, {input_name: X})[0]
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print(confusion_matrix(y_true, y_pred_classes))
    print(classification_report(y_true, y_pred_classes, target_names=class_to_idx.keys()))

    df = pd.DataFrame({
        'image_name': names,
        'true_class': y_true,
        'pred_class': y_pred_classes
    })

    df.to_csv(os.path.join(output_dir, 'ONNX16_predictions.csv'), index=False)

    return df

start = time.perf_counter()
run_onnx_prediction()
end = time.perf_counter()

print(f"Eltelt idő: {end-start}")
