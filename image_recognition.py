import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. 基本設定 ---
# ★★★ あなたの環境に合わせて、データセットへのパスを必ず変更してください ★★★
# 例: 'C:/Users/YourUser/Desktop/ImageRecognition/train_images'
train_data_dir = 'C:/Users/komai/OneDrive/デスクトップ/プロコンに使えそうなやつ/ImageRecognition/train_images'
validation_data_dir = 'C:/Users/komai/OneDrive/デスクトップ/プロコンに使えそうなやつ/ImageRecognition/test_images'

# 画像のサイズとバッチサイズ
img_rows, img_cols = 224, 224
channels = 3
nb_batch_size = 32
nb_epoch = 30 # 学習のエポック数

# --- 2. GPUのメモリ設定 (NVIDIA GPUがある場合) ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Running on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)
else:
    print("Running on CPU.")

# --- 3. データの前処理とジェネレータの作成 ---
# 学習データ用のジェネレータ（データ拡張あり）
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255, # VGG16の前処理に合わせてrescaleを削除し、preprocess_inputで行う
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    preprocessing_function=keras.applications.vgg16.preprocess_input # VGG16用の前処理を追加
)

# 検証データ用のジェネレータ（データ拡張なし）
val_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    preprocessing_function=keras.applications.vgg16.preprocess_input # VGG16用の前処理を追加
)

# ジェネレータを作成
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=nb_batch_size,
    class_mode='categorical',
    shuffle=True # 学習時はシャッフルするのが一般的
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=nb_batch_size,
    class_mode='categorical',
    shuffle=False
)

# クラスの数を取得
nb_classes = len(train_generator.class_indices)

# --- 4. モデルの構築 (VGG16をベースにした転移学習) ---

# VGG16モデルをロード（トップの全結合層は含めない）
base_model = keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(img_rows, img_cols, channels)
)

# ベースモデルの層を凍結（学習させないようにする）
base_model.trainable = False

# 新しいモデルを定義
model = keras.models.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(nb_classes, activation='softmax')
])

# モデルのコンパイル
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), # Adamが一般的
    metrics=['accuracy']
)

model.summary()

# --- 5. モデルの学習または重みの読み込み ---
# 保存する重みファイルの名前を定義
weights_filename = f"weight_transfer_learning_{nb_classes}classes.weights.h5"

# もし学習済みの重みファイルが存在すれば、それを読み込む
if os.path.exists(weights_filename):
    print(f"Found existing weights file: {weights_filename}")
    print("Skipping training and loading weights.")
    model.load_weights(weights_filename)
    print("Weights loaded successfully.")

# もし重みファイルがなければ、学習を実行する
else:
    print("No weights file found. Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // nb_batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // nb_batch_size,
        verbose=1
    )
    # 学習済みモデルの重みを保存
    model.save_weights(weights_filename)
    print(f"Training finished and weights saved to {weights_filename}.")

    # --- 6. 学習結果の可視化 ---
    def plot_history(history):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 精度のプロット
        axes[0].set_title("Accuracy vs Epoch")
        axes[0].plot(history.history["accuracy"], label="Training")
        axes[0].plot(history.history["val_accuracy"], label="Validation")
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # 損失のプロット
        axes[1].set_title("Loss vs Epoch")
        axes[1].plot(history.history["loss"], label="Training")
        axes[1].plot(history.history["val_loss"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    plot_history(history)

# --- 7. 画像認識の実行 ---
# クラス名とインデックスの対応辞書を作成
inv_class_dict = {v: k for k, v in train_generator.class_indices.items()}

def predict_picture(img_path):
    # 画像のロードと前処理
    img = keras.preprocessing.image.load_img(img_path, target_size=(img_rows, img_cols))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x) # VGG16用の前処理

    # 予測
    pred = model.predict(x)[0]

    # 結果の表示
    plt.imshow(np.array(img))
    ax = plt.gca()
    ax.grid(color='w', linestyle='none')
    plt.show()

    # 予測確率が高いトップ5を出力
    top = 5
    top_indices = pred.argsort()[-top:][::-1]
    result = [(inv_class_dict[i], pred[i]) for i in top_indices]
    
    print(f"Prediction for {os.path.basename(img_path)}:")
    for class_name, probability in result:
        print(f"  - {class_name}: {probability:.4f}")

# ★★★ 実際に予測したい画像のパスを指定してください ★★★
# 以下のコードは、検証用フォルダの最初のクラスにある最初の画像を自動で予測します。
# 別の画像を試したい場合は、下の 'test_image_to_predict' の行を書き換えてください。
try:
    # 検証ディレクトリ内の最初のクラスフォルダを取得
    first_class_folder = os.path.join(validation_data_dir, os.listdir(validation_data_dir)[0])
    if os.path.isdir(first_class_folder):
        # そのクラスフォルダ内の最初の画像ファイルを取得
        # first_image_file = os.listdir(first_class_folder)[0]
        # test_image_to_predict = os.path.join(first_class_folder, first_image_file)
        #この下のプログラムを変更する
        test_image_to_predict = os.path.join(r"C:\Users\komai\OneDrive\デスクトップ\プロコンに使えそうなやつ\ImageRecognition\test_images\seal_Green\20250812_153513_aug_1.jpg".replace("\\", "/"))
        print(f"\n--- Automatically predicting the first test image: {test_image_to_predict} ---")
        predict_picture(test_image_to_predict)
    else:
        print("\n--- Could not find a valid class folder in the validation directory to predict. ---")

except (FileNotFoundError, IndexError):
    print("\n--- Could not find any image to predict in the validation directory. ---")
    print("--- Please check if your validation data directory is set correctly and contains images. ---")


