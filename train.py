import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import os
from loguru import logger

from model import (
    CNN3D,
    CNN3DWithMobileAttention,
    CNN3DWithPoolFormer,
    CNN3DWithMobileAttentionAndPE,
    CNN3DWithPoolFormerAndPE,
    R2plus1d,
    R2plus1dWithMobileAttention,
    R2plus1dWithPoolFormer,
    R2plus1dWithMobileAttentionAndPE,
    R2plus1dWithPoolFormerAndPE,
    I3D,
    I3DWithMobileAttention,
    I3DWithPoolFormer
)
from uniformerv2_model import uniformerv2_tiny, uniformerv2_nano

# --- モデル情報の出力機能 ---
@logger.catch
def print_model_info(model, input_shape, model_class_name):
    """
    モデルのパラメータ数とFLOPsを計算して表示する関数

    Args:
        model: 分析するモデル
        input_shape: 入力テンソルの形状 (batch_size, channels, frames, height, width)
    """
    # パラメータ数を計算
    param_count = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {param_count:,}")

    # 演算量(FLOPs)を計算
    dummy_input = torch.rand(input_shape)
    dummy_input = dummy_input.to(next(model.parameters()).device)

    try:
        flops = FlopCountAnalysis(model, dummy_input)
        flops_table = flop_count_table(flops)
        total_flops = flops.total()

        print("--- モデル演算量(FLOPs)の詳細 ---")
        print(flops_table)
        print(f"総演算量: {total_flops / 1e9:.2f} GFLOPs")

        # 結果をファイルに保存
        os.makedirs("model_analysis", exist_ok=True)
        with open(f"model_analysis/{model_class_name}_info.txt", "w") as f:
            f.write(f"モデル名: {model_class_name}\n")
            f.write(f"パラメータ数: {param_count:,}\n")
            f.write(f"入力形状: {input_shape}\n")
            f.write(f"総演算量: {total_flops / 1e9:.2f} GFLOPs\n\n")
            f.write("--- 演算量の詳細 ---\n")
            f.write(flops_table)

        print(f"モデル情報を model_analysis/{model_class_name}_info.txt に保存しました")

        return {
            "param_count": param_count,
            "total_flops": total_flops,
            "flops_giga": total_flops / 1e9
        }

    except Exception as e:
        print(f"FLOPs計算中にエラーが発生しました: {e}")
        return {
            "param_count": param_count,
            "total_flops": None,
            "flops_giga": None
        }


# --- 疑似データ生成コード ---
@logger.catch
def generate_dummy_data(batch_size, num_frames, height, width, num_classes):
    """
    疑似動画行動分離データを生成する関数

    Args:
        batch_size (int): バッチサイズ
        num_frames (int): 1つの動画クリップのフレーム数 (Depth)
        height (int): フレームの高さ
        width (int): フレームの幅
        num_classes (int): クラス数

    Returns:
        tuple: (疑似入力データ tensor, 疑似ラベル tensor)
    """
    # 入力データの形状: (batch_size, channels=3, num_frames, height, width)
    # channels=3はRGBを想定
    dummy_input = torch.randn(batch_size, 3, num_frames, height, width)

    # ラベルの形状: (batch_size,) - クラスID (0 から num_classes-1)
    # 分類問題なので、ターゲットはクラスIDとなります
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    return dummy_input, dummy_labels

def quantize_model(model):
    """モデルを量子化 (int8)"""
    try:
        # GPUモデルをCPUに移動（量子化はCPUでのみ実行可能）
        model = model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv3d},
            dtype=torch.qint8
        )
        return quantized_model
    except Exception as e:
        raise Exception(f"An error occurred during model quantization: {str(e)}")


# モデルのパラメータ設定
INPUT_FRAMES = 5
IMAGE_HEIGHT = 120  # 例として適切なサイズを設定
IMAGE_WIDTH = 180  # 例として適切なサイズを設定
NUM_CLASSES = 7
BATCH_SIZE = 4  # 学習可能なように小さいバッチサイズに設定
NUM_EPOCHS = 10  # テスト用にエポック数を少なく設定
IS_UNIFORMER = False  # UniformerV2を使用するかどうかのフラグ



def train():
    # モデルのインスタンス化
    if IS_UNIFORMER:
        model = uniformerv2_nano(
            t_size=INPUT_FRAMES,
            height=120,
            width=180,
            patch_size=12  # パッチサイズ12: 120/12=10, 180/12=15
        )
        model_class_name = "uniformerv2_nano"
    else:
        model = R2plus1d(in_channels=3, num_classes=NUM_CLASSES)
        model_class_name = model.__class__.__name__.lower()

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()  # 分類問題なので交差エントロピー損失を使用
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adamオプティマイザを使用

    # 疑似データの生成 (学習ループの外で一度だけ生成)
    # 実際の学習では、データローダーを使用してデータセットからバッチを読み込みます
    dummy_input, dummy_labels = generate_dummy_data(BATCH_SIZE, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)

    # 学習ループ
    print("--- 学習開始 ---")
    for epoch in range(NUM_EPOCHS):
        # モデルを訓練モードに設定
        model.train()

        # 疑似データをモデルに入力し、出力を得る
        outputs = model(dummy_input)

        # 損失を計算
        loss = criterion(outputs, dummy_labels)

        # 勾配をゼロクリア、バックプロパゲーション、パラメータ更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 進捗を表示 (例: 1エポックごとに損失を表示)
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
    print("--- 学習終了 ---")

    # モデルを評価モードに設定
    model.eval()
    with torch.no_grad():  # 勾配計算を無効化
        # 推論用の疑似データを生成 (例: バッチサイズ 1)
        inference_input, _ = generate_dummy_data(1, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES)
        inference_output = model(inference_input)
        predicted_class = torch.argmax(inference_output, dim=1)
        print(f"推論結果 (予測クラスID): {predicted_class.item()}")

    # モデルをトレースする前にFLOPs分析を実行
    print("\n--- モデルの分析開始 ---")
    input_shape = (BATCH_SIZE, 3, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
    model_info = print_model_info(model, input_shape, model_class_name)
    print("--- モデル分析完了 ---\n")

    # モデルのフォワードパスの確認 (疑似データを使用)
    # TorchScriptでトレースするために、モデルが期待する形状の入力が必要です
    dummy_input_for_trace = torch.randn(BATCH_SIZE, 3, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
    # モデルがGPUにある場合は、入力データもGPUに移動させてください
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input_for_trace = dummy_input_for_trace.to(device)

    try:
        model.eval()  # モデルを評価モードにすることで、トレーニング特有の演算（例: dropout）を無効化
        # モデルをTorchScript形式でトレース
        print("--- モデルをTorchScriptでトレース中 ---")
        traced_model = torch.jit.trace(model, dummy_input_for_trace)
        print("--- トレース完了 ---")

        # TorchScriptモデルをファイルに保存
        save_path = f"models/{model_class_name}.pt"
        traced_model.save(save_path)
        print(f"モデルを {save_path} に保存しました。")

    except Exception as e:
        print(f"TorchScriptのトレースまたは保存中にエラーが発生しました: {e}")
        print("モデルにTorchScriptでトレースできない演算が含まれている可能性があります。")
        print("詳細については、PyTorchのTorchScriptのドキュメントを参照してください。")

    # オプション: 保存したTorchScriptモデルをPyTorchでロードしてテスト
    try:
        print(f"--- {save_path} からモデルをロード中 ---")
        loaded_model = torch.jit.load(save_path)
        print("--- ロード完了 ---")

        # ロードしたモデルで推論を実行
        # ロードしたモデルがGPUにある場合は、入力データもGPUに移動させてください
        inference_input_loaded = dummy_input_for_trace.to(
            device) if torch.cuda.is_available() else dummy_input_for_trace
        loaded_output = loaded_model(inference_input_loaded)
        print("ロードしたモデルでの出力形状:", loaded_output.shape)
        print("ロードしたモデルでの推論テスト成功。")

    except Exception as e:
        print(f"保存したモデルのロードまたはテスト中にエラーが発生しました: {e}")


    # --- モデルをONNX形式で保存 ---
    print("--- モデルをONNX形式でエクスポート中 ---")
    try:
        onnx_save_path = f"models/{model_class_name}.onnx"

        # ONNX形式でモデルをエクスポート
        torch.onnx.export(model,  # 実行するモデル
                          dummy_input_for_trace,  # モデルの入力となるテンソル
                          onnx_save_path,  # モデルを保存するファイルパス
                          export_params=True,  # 学習済みパラメータをエクスポートファイルに含める
                          opset_version=11,  # ONNX opset バージョン (適宜変更)
                          do_constant_folding=True,  # 定数畳み込みを許可
                          input_names=['input'],  # モデルの入力テンサルの名前
                          output_names=['output'],  # モデルの出力テンサルの名前
                          # dynamic_axes={'input' : {0 : 'batch_size'},    # 可変長のバッチサイズをサポートする場合
                          #               'output' : {0 : 'batch_size'}})
                          )

        print(f"モデルを {onnx_save_path} に保存しました。")

        print("--- ONNXエクスポート完了 ---")

    except Exception as e:
        print(f"ONNX形式でのエクスポート中にエラーが発生しました: {e}")
        print("モデルにONNXエクスポートに対応していない演算が含まれている可能性があります。")
        print("または、ONNX opsetバージョンが適切でない可能性があります。")
        print("詳細については、PyTorchのONNXエクスポートのドキュメントを参照してください。")



    # 量子化されたモデルで推論
    quantized_model = quantize_model(model)

    try:
        cpu_dummy_input = dummy_input_for_trace.cpu()

        quantized_model.eval()  # モデルを評価モードにすることで、トレーニング特有の演算（例: dropout）を無効化
        # モデルをTorchScript形式でトレース
        print("--- モデルをTorchScriptでトレース中 ---")
        traced_model = torch.jit.trace(quantized_model, cpu_dummy_input)
        print("--- トレース完了 ---")

        # TorchScriptモデルをファイルに保存
        save_path = f"models/{model_class_name}_int8.pt"
        traced_model.save(save_path)
        print(f"量子化モデルを {save_path} に保存しました。")

    except Exception as e:
        print(f"量子化モデルのトレースまたは保存中にエラーが発生しました: {e}")
        print("詳細については、PyTorchのTorchScriptのドキュメントを参照してください。")


if __name__ == "__main__":
    train()
