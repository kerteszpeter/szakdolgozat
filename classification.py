import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def make_model(num_classes, base_model='NASNet'):
    if base_model == 'NASNet':
        base = tf.keras.applications.NASNetLarge(weights=None, include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported base_model")

    base.trainable = True
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

def rename_classes(test_dict):
    img_cnt = sum(test_dict.values())
    return {
        'img_cnt': img_cnt,
        'healthy_cnt': test_dict.get(0, 0),
        'rubbish_cnt': test_dict.get(1, 0),
        'unhealthy_cnt': test_dict.get(2, 0)
    }

def run_nasnet_prediction(input_dir='input1', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idx = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}

    images, names, y_true = [], [], []

    for class_name, class_idx in class_to_idx.items():
        for fname in os.listdir(os.path.join(input_dir, class_name)):
            img = Image.open(os.path.join(input_dir, class_name, fname)).convert("RGB").resize((224, 224))
            images.append(np.array(img, dtype="uint8") / 255.0)
            names.append(fname)
            y_true.append(class_idx)

    X = np.stack(images)

    test_flow = ImageDataGenerator().flow(x=X, shuffle=False)

    model = make_model(3, base_model='NASNet')
    model.load_weights('models/NasNetLarge/raw_weights.h5')

    y_pred = model.predict(test_flow, verbose=False)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Accuracy:", accuracy_score(y_true, y_pred_classes))
    print(confusion_matrix(y_true, y_pred_classes))
    print(classification_report(y_true, y_pred_classes, target_names=class_to_idx.keys()))

    df = pd.DataFrame({
        'image_name': names,
        'true_class': y_true,
        'pred_class': y_pred_classes
    })

    df.to_csv(os.path.join(output_dir, 'NasNetLarge_prediction.csv'), index=False)

    return df

start = time.perf_counter() ###
run_nasnet_prediction()
end = time.perf_counter() ###

print(f"Eltelt idő: {end-start}")
"""
model = make_model(3, base_model='NASNet')
model.load_weights('models/NasNetLarge/raw_weights.h5')
model.save("models/NasNetLarge/saved_model")
"""
