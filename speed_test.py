import glob
import time
import numpy as np
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
    models = glob.glob("models/*.pt")
    # 結果を格納するためのリスト
    results = []

    # 量子化モデル用とその他のモデル用で分岐できるように判定する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"実行デバイス: {device}")

    dummy_input = torch.randn(BATCH_SIZE, 3, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH)

    for save_path in models:
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

        # 各点にモデル名のラベルをつける
        plt.scatter(df['avg_inference_time'], df['param_count_millions'], s=100, alpha=0.7)

        for i, txt in enumerate(df['model_name']):
            model_name_short = txt.replace('.pt', '')
            plt.annotate(model_name_short,
                         (df['avg_inference_time'].iloc[i], df['param_count_millions'].iloc[i]),
                         xytext=(5, 5), textcoords='offset points')

        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Parameter Count (M)')
        plt.title('Model Parameters vs. Inference Time')
        plt.grid(True, linestyle='--', alpha=0.7)

        # エラーバーを追加（水平方向のみ）
        plt.errorbar(df['avg_inference_time'], df['param_count_millions'],
                     xerr=df['std_inference_time'], fmt='none', ecolor='gray', capsize=5)

        plt.tight_layout()
        plt.savefig('model_performance_comparison.png')
        print("グラフをmodel_performance_comparison.pngに保存しました")
        plt.show()
    else:
        print("結果がありません。モデルのロードに失敗した可能性があります。")


if __name__ == "__main__":
    main()