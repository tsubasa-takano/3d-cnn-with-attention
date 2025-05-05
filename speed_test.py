import glob
import time
import numpy as np
import onnxruntime as ort
import onnx
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

# モデルのパラメータ設定
INPUT_FRAMES = 5
IMAGE_HEIGHT = 120  # 例として適切なサイズを設定
IMAGE_WIDTH = 180  # 例として適切なサイズを設定
NUM_CLASSES = 7
BATCH_SIZE = 1
NUM_WARMUP = 5  # ウォームアップ回数
NUM_RUNS = 1000  # 計測回数


def get_model_size_mb(model_path):
    """モデルファイルのサイズをMB単位で返す"""
    return os.path.getsize(model_path) / (1024 * 1024)


def main():
    # modelsフォルダ内のモデルをすべて取得
    models_pt = glob.glob("models/*.pt")
    models_onnx = glob.glob("models/*.onnx")
    # 結果を格納するためのリスト
    results = []

    # 量子化モデル用とその他のモデル用で分岐できるように判定する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"実行デバイス: {device}")

    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    for save_path in models_pt:
        model_name = os.path.basename(save_path)
        print(f"\n--- {model_name} の処理速度とサイズを計測中 ---")

        try:
            # モデルをロード
            loaded_model = torch.jit.load(save_path)

            # 量子化モデルかどうかを判断（単純にファイル名に "int8" が含まれるかで判定）
            is_quantized = "int8" in model_name

            # 量子化モデルは CPU で実行、それ以外は指定されたデバイスで実行
            model_device = torch.device("cpu") if is_quantized else device
            loaded_model = loaded_model.to(model_device)
            loaded_model.eval()

            # 入力テンソルをモデルと同じデバイスに移動
            input_tensor = dummy_input.to(model_device)

            # モデルのパラメータ数を取得
            param_count = sum(p.numel() for p in loaded_model.parameters())
            print(f"パラメータ数: {param_count:,}")
            print(f"実行デバイス: {model_device}")

            # ウォームアップ実行
            print("ウォームアップ実行中...")
            with torch.no_grad():
                for _ in range(NUM_WARMUP):
                    _ = loaded_model(input_tensor)

            # 処理時間の計測
            print(f"{NUM_RUNS}回の推論を実行中...")
            inference_times = []
            with torch.no_grad():
                for i in range(NUM_RUNS):
                    start_time = time.time()
                    _ = loaded_model(input_tensor)

                    # GPUの場合は同期を取る
                    if model_device.type == 'cuda':
                        torch.cuda.synchronize()

                    end_time = time.time()
                    inference_time = (end_time - start_time) * 1000  # ミリ秒単位
                    inference_times.append(inference_time)

            # 平均と標準偏差を計算
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)

            print(f"平均推論時間: {avg_time:.2f} ms ± {std_time:.2f} ms")

            # 結果をリストに追加
            results.append({
                'model_name': model_name,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'param_count': param_count,
                'device': str(model_device)
            })

        except Exception as e:
            print(f"エラーが発生しました: {e}")

    for save_path in models_onnx:
        model_name = os.path.basename(save_path)
        print(f"\n--- {model_name} の処理速度とサイズを計測中 ---")

        try:
            # モデルサイズを取得
            model_size_mb = get_model_size_mb(save_path)
            print(f"モデルサイズ: {model_size_mb:.2f} MB")

            # ONNXモデル処理部分で以下のように使用
            param_count = get_onnx_param_count(save_path)
            print(f"パラメータ数: {param_count:,}")

            # セッション作成（CUDA対応のONNXRuntimeがインストールされている場合はGPUを使用）
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(save_path, providers=providers)

            # モデルの入力情報を取得
            input_info = session.get_inputs()[0]
            print(f"入力名: {input_info.name}")
            print(f"入力形状: {input_info.shape}")
            print(f"入力データ型: {input_info.type}")

            # 実際に使用されているプロバイダを確認
            used_provider = session.get_providers()[0]
            device_name = "GPU" if "CUDA" in used_provider else "CPU"
            print(f"実行デバイス: {device_name}")

            input_name = input_info.name

            ort_input = {input_name: dummy_input.numpy()}

            # ウォームアップ実行
            print("ウォームアップ実行中...")
            for _ in range(NUM_WARMUP):
                _ = session.run(None, ort_input)

            # 処理時間の計測
            print(f"{NUM_RUNS}回の推論を実行中...")
            inference_times = []
            for i in range(NUM_RUNS):
                start_time = time.time()
                _ = session.run(None, ort_input)
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ミリ秒単位
                inference_times.append(inference_time)

            # 平均と標準偏差を計算
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)

            print(f"平均推論時間: {avg_time:.2f} ms ± {std_time:.2f} ms")

            # パラメータ数はONNXモデルから直接取得できないため、ファイルサイズを参考値として使用
            results.append({
                'model_name': model_name,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'param_count': param_count,  # ONNXからは直接取得できない
                'model_size_mb': model_size_mb,
                'device': device_name
            })

        except Exception as e:
            print(f"ONNXモデル処理中にエラーが発生しました: {e}")

    # 結果をpandasデータフレームに変換
    if results:
        df = pd.DataFrame(results)

        # パラメータ数をM（百万）単位に変換
        df['param_count_millions'] = df['param_count'] / 1_000_000

        print("\n--- 結果サマリー ---")
        print(df)

        # CSVに保存
        df.to_csv('model_performance_results.csv', index=False)
        print("結果をmodel_performance_results.csvに保存しました")

        # 散布図の作成
        plt.figure(figsize=(10, 8))

        # PyTorchとONNXモデルを区別するためのマーカー設定
        for i, row in df.iterrows():
            if '.pt' in row['model_name']:
                marker = 'o'  # PyTorchモデル用
                color = 'blue'
            else:
                marker = 's'  # ONNXモデル用
                color = 'red'

            # パラメータ数がNoneの場合はモデルサイズを使用
            y_value = row['param_count_millions'] if row['param_count'] is not None else row['model_size_mb'] / 10

            plt.scatter(row['avg_inference_time'], y_value, s=100, alpha=0.7, marker=marker, color=color)

            model_name_short = row['model_name']
            plt.annotate(model_name_short + (' (ONNX)' if '.onnx' in row['model_name'] else ''),
                        (row['avg_inference_time'], y_value),
                        xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Parameter Count (M)')
        plt.title('Model Parameters vs. Inference Time')
        plt.grid(True, linestyle='--', alpha=0.7)

        # エラーバーを追加
        plt.errorbar(df['avg_inference_time'],
                     df.apply(lambda x: x['param_count_millions'] if x['param_count'] is not None else x['model_size_mb']/10, axis=1),
                     xerr=df['std_inference_time'], fmt='none', ecolor='gray', capsize=5)
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png')
        print("グラフをmodel_performance_comparison.pngに保存しました")
        plt.show()
    else:
        print("結果がありません。モデルのロードに失敗した可能性があります。")


def get_onnx_param_count(model_path):
    """
    ONNXモデルのパラメータ総数を計算する関数

    Args:
        model_path: ONNXモデルのファイルパス

    Returns:
        int: パラメータ総数
    """
    # ONNXモデルをロード
    model = onnx.load(model_path)

    # 初期化されたテンソル（学習済みパラメータ）を取得
    param_count = 0
    for initializer in model.graph.initializer:
        # 各テンソルの形状からパラメータ数を計算
        # dimsはdim_valueを持つオブジェクトか、直接整数値の場合がある
        shape = []
        for dim in initializer.dims:
            if hasattr(dim, 'dim_value'):
                shape.append(dim.dim_value)
            else:
                # 直接整数値の場合
                shape.append(dim)

        num_params = np.prod(shape)
        param_count += num_params

    return int(param_count)  # 整数値として返す

if __name__ == "__main__":
    main()