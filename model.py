import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import *

# 基本的な3D-CNNモデル (アテンションモジュールを組み込む)
class CNN3DWithMobileAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(CNN3DWithMobileAttention, self).__init__()

        # 3D CNN層 (特徴量を抽出)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        # アテンションを適用する手前ではプーリングで空間サイズを調整
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Mobile-MQA風3Dアテンションモジュール
        # conv3の出力チャネル数128をattentionモジュールの入力とする
        # num_heads, key_dim, value_dimはMobileNetV4のHybridモデルのMQA設定を参考に
        # (例: mobilenetv4_hybrid_largeではh=8, d=64など)
        self.attention_module = MobileAttention3D(in_channels=128, num_heads=8, key_dim=64, value_dim=64)

        # 全結合層による分類ヘッド
        # アテンションモジュールの出力後の特徴マップをプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes) # Attentionモジュールの出力チャネル数は入力と同じにしている

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, 3, 5, H, W) を想定

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # アテンションモジュールを適用
        x = self.attention_module(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



# 基本的な3D-CNNモデル (PoolFormerブロックを組み込む)
class CNN3DWithPoolFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pool_size=(1, 3, 3), mlp_ratio=4.):
        super(CNN3DWithPoolFormer, self).__init__()

        # 3D CNN層 (特徴量を抽出) - ここは元のモデルと同じ構造を維持
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        # PoolFormerブロックの前に空間・時間サイズを調整するためのプーリング
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # MobileAttention3D を PoolFormer3DBlock に置き換え
        # conv3/pool3 の出力チャネル数128を入力とする
        self.poolformer_block = PoolFormer3DBlock(dim=128, pool_size=(1, 3, 3), mlp_ratio=4.)

        # 全結合層による分類ヘッド
        # PoolFormerブロックの出力後の特徴マップをグローバルプーリングしてフラット化
        # PoolFormerブロックは空間サイズを変更しないので、GlobalAvgPool3dの出力チャネルは入力と同じ128
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # PoolFormerブロックを適用
        x = self.poolformer_block(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CNN3DWithMobileAttentionAndPE(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(CNN3DWithMobileAttentionAndPE, self).__init__()

        # 3D CNN層 (特徴量を抽出)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        # アテンションを適用する手前ではプーリングで空間サイズを調整
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # アテンションモジュールに入力される特徴マップの想定サイズに基づいてPosition Embedding層を定義
        # ここでの D, H, W はプーリング後のサイズに合わせて静的に定義するか、
        # Dynamic Padding などで対応する必要がある場合があります。
        # 例として、もしアテンション入力が (B, 128, 4, 14, 14) なら:
        self.position_embedding = PositionEmbedding3D(channels=128, depth=4, height=15,
                                                      width=22)  # <- Position Embedding層を追加

        # Mobile-MQA風3Dアテンションモジュール
        # conv3の出力チャネル数128をattentionモジュールの入力とする
        # num_heads, key_dim, value_dimはMobileNetV4のHybridモデルのMQA設定を参考に
        # (例: mobilenetv4_hybrid_largeではh=8, d=64など)
        self.attention_module = MobileAttention3D(in_channels=128, num_heads=8, key_dim=64, value_dim=64)

        # 全結合層による分類ヘッド
        # アテンションモジュールの出力後の特徴マップをプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes) # Attentionモジュールの出力チャネル数は入力と同じにしている

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, 3, 5, H, W) を想定

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # アテンションモジュールを適用
        x = self.attention_module(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 基本的な3D-CNNモデル (PoolFormerブロックを組み込む)
class CNN3DWithPoolFormerAndPE(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pool_size=(1, 3, 3), mlp_ratio=4.):
        super(CNN3DWithPoolFormerAndPE, self).__init__()

        # 3D CNN層 (特徴量を抽出) - ここは元のモデルと同じ構造を維持
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        # PoolFormerブロックの前に空間・時間サイズを調整するためのプーリング
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # アテンションモジュールに入力される特徴マップの想定サイズに基づいてPosition Embedding層を定義
        # ここでの D, H, W はプーリング後のサイズに合わせて静的に定義するか、
        # Dynamic Padding などで対応する必要がある場合があります。
        # 例として、もしアテンション入力が (B, 128, 4, 14, 14) なら:
        self.position_embedding = PositionEmbedding3D(channels=128, depth=4, height=15,
                                                      width=22)  # <- Position Embedding層を追加

        # MobileAttention3D を PoolFormer3DBlock に置き換え
        # conv3/pool3 の出力チャネル数128を入力とする
        self.poolformer_block = PoolFormer3DBlock(dim=128, pool_size=(1, 3, 3), mlp_ratio=4.)

        # 全結合層による分類ヘッド
        # PoolFormerブロックの出力後の特徴マップをグローバルプーリングしてフラット化
        # PoolFormerブロックは空間サイズを変更しないので、GlobalAvgPool3dの出力チャネルは入力と同じ128
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # PoolFormerブロックを適用
        x = self.poolformer_block(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# R(2+1)DとMobileAttentionを組み合わせた3D-CNNモデル
class R2plus1dWithMobileAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(R2plus1dWithMobileAttention, self).__init__()

        # R(2+1)D分解された畳み込みブロック
        # 各ブロックでパラメータ数を減らしつつ特徴抽出
        # 元のconv層の kernel_size=3, padding=1 を R(2+1)Dブロックに置き換える

        # ブロック1: 3->32 ch
        # カーネルサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block1 = R2plus1dBlock(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ブロック2: 32->64 ch
        # カーnelサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block2 = R2plus1dBlock(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ブロック3: 64->128 ch
        # カーネルサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block3 = R2plus1dBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # アテンションを適用する手前ではプーリングで空間サイズを調整
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Mobile-MQA風3Dアテンションモジュール
        # r2plus1d_block3の出力チャネル数128をattentionモジュールの入力とする
        self.attention_module = MobileAttention3D(in_channels=128, num_heads=8, key_dim=64, value_dim=64)

        # 全結合層による分類ヘッド
        # アテンションモジュールの出力後の特徴マップをプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes) # Attentionモジュールの出力チャネル数は入力と同じにしている

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, 3, D, H, W)

        # R(2+1)Dブロックとプーリングを順に適用
        x = self.pool1(self.r2plus1d_block1(x))
        x = self.pool2(self.r2plus1d_block2(x))
        x = self.pool3(self.r2plus1d_block3(x))

        # アテンションモジュールを適用
        x = self.attention_module(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# R(2+1)DブロックとPoolFormerブロックを組み合わせた3D-CNNモデル
class R2plus1dWithPoolFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pool_size=(1, 3, 3), mlp_ratio=4.):
        super(R2plus1dWithPoolFormer, self).__init__()

        # R(2+1)D分解された畳み込みブロックに置き換え
        # 元のConv3d + BN + ReLU のシーケンスを R2plus1dBlock で置き換え
        # 各ブロックのstride=(1, 1, 1)は、後続のMaxPoolでダウンサンプリングを行うため

        # ブロック1: 3->32 ch
        # 元の Conv3d(3, 32, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block1 = R2plus1dBlock(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # 空間次元のみをダウンサンプリングするプーリング層
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ブロック2: 32->64 ch
        # 元の Conv3d(32, 64, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block2 = R2plus1dBlock(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # 時間次元と空間次元をダウンサンプリングするプーリング層
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ブロック3: 64->128 ch
        # 元の Conv3d(64, 128, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block3 = R2plus1dBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # PoolFormerブロックを適用する手前で空間・時間サイズを調整するためのプーリング
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # PoolFormer3DBlock (以前のコードと同じものを使用)
        # 入力チャネルは最後のR2plus1dブロックの出力 (128)
        # pool_size=(1, 3, 3) は以前の修正で適用済み
        self.poolformer_block = PoolFormer3DBlock(dim=128, pool_size=(1, 3, 3), mlp_ratio=4.)

        # 全結合層による分類ヘッド (以前のコードと同じものを維持)
        # PoolFormerブロックの出力後の特徴マップをグローバルプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        # R(2+1)Dブロックとそれに続くプーリング層を適用
        x = self.r2plus1d_block1(x)
        x = self.pool1(x)
        # print("Shape after R2+1D block1/pool1:", x.shape) # デバッグ用

        x = self.r2plus1d_block2(x)
        x = self.pool2(x)
        # print("Shape after R2+1D block2/pool2:", x.shape) # デバッグ用

        x = self.r2plus1d_block3(x)
        x = self.pool3(x)
        # print("Shape before PoolFormer block:", x.shape) # デバッグ用

        # PoolFormerブロックを適用
        x = self.poolformer_block(x)
        # print("Shape after PoolFormer block:", x.shape) # デバッグ用

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# R(2+1)DとMobileAttentionを組み合わせた3D-CNNモデル
class R2plus1dWithMobileAttentionAndPE(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(R2plus1dWithMobileAttentionAndPE, self).__init__()

        # R(2+1)D分解された畳み込みブロック
        # 各ブロックでパラメータ数を減らしつつ特徴抽出
        # 元のconv層の kernel_size=3, padding=1 を R(2+1)Dブロックに置き換える

        # ブロック1: 3->32 ch
        # カーネルサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block1 = R2plus1dBlock(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ブロック2: 32->64 ch
        # カーnelサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block2 = R2plus1dBlock(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ブロック3: 64->128 ch
        # カーネルサイズ(3, 3, 3), パディング(1, 1, 1), ストライド(1, 1, 1) 相当
        self.r2plus1d_block3 = R2plus1dBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # アテンションを適用する手前ではプーリングで空間サイズを調整
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # アテンションモジュールに入力される特徴マップの想定サイズに基づいてPosition Embedding層を定義
        # ここでの D, H, W はプーリング後のサイズに合わせて静的に定義するか、
        # Dynamic Padding などで対応する必要がある場合があります。
        # 例として、もしアテンション入力が (B, 128, 4, 14, 14) なら:
        self.position_embedding = PositionEmbedding3D(channels=128, depth=4, height=15,
                                                      width=22)  # <- Position Embedding層を追加

        # Mobile-MQA風3Dアテンションモジュール
        # r2plus1d_block3の出力チャネル数128をattentionモジュールの入力とする
        self.attention_module = MobileAttention3D(in_channels=128, num_heads=8, key_dim=64, value_dim=64)

        # 全結合層による分類ヘッド
        # アテンションモジュールの出力後の特徴マップをプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes) # Attentionモジュールの出力チャネル数は入力と同じにしている

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, 3, D, H, W)

        # R(2+1)Dブロックとプーリングを順に適用
        x = self.pool1(self.r2plus1d_block1(x))
        x = self.pool2(self.r2plus1d_block2(x))
        x = self.pool3(self.r2plus1d_block3(x))

        # Position Embedding を適用 <- ここに追加
        # Note: PositionEmbedding3D のインスタンス化時に指定した D, H, W と
        # ここでの x の D, H, W が一致している必要があります。
        x = self.position_embedding(x)

        # アテンションモジュールを適用
        x = self.attention_module(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# R(2+1)DブロックとPoolFormerブロックを組み合わせた3D-CNNモデル
class R2plus1dWithPoolFormerAndPE(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pool_size=(1, 3, 3), mlp_ratio=4.):
        super(R2plus1dWithPoolFormerAndPE, self).__init__()

        # R(2+1)D分解された畳み込みブロックに置き換え
        # 元のConv3d + BN + ReLU のシーケンスを R2plus1dBlock で置き換え
        # 各ブロックのstride=(1, 1, 1)は、後続のMaxPoolでダウンサンプリングを行うため

        # ブロック1: 3->32 ch
        # 元の Conv3d(3, 32, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block1 = R2plus1dBlock(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # 空間次元のみをダウンサンプリングするプーリング層
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ブロック2: 32->64 ch
        # 元の Conv3d(32, 64, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block2 = R2plus1dBlock(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # 時間次元と空間次元をダウンサンプリングするプーリング層
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ブロック3: 64->128 ch
        # 元の Conv3d(64, 128, kernel_size=3, padding=1, stride=1) に相当
        self.r2plus1d_block3 = R2plus1dBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        # PoolFormerブロックを適用する手前で空間・時間サイズを調整するためのプーリング
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # アテンションモジュールに入力される特徴マップの想定サイズに基づいてPosition Embedding層を定義
        # ここでの D, H, W はプーリング後のサイズに合わせて静的に定義するか、
        # Dynamic Padding などで対応する必要がある場合があります。
        # 例として、もしアテンション入力が (B, 128, 4, 14, 14) なら:
        self.position_embedding = PositionEmbedding3D(channels=128, depth=4, height=15,
                                                      width=22)  # <- Position Embedding層を追加

        # PoolFormer3DBlock (以前のコードと同じものを使用)
        # 入力チャネルは最後のR2plus1dブロックの出力 (128)
        # pool_size=(1, 3, 3) は以前の修正で適用済み
        self.poolformer_block = PoolFormer3DBlock(dim=128, pool_size=(1, 3, 3), mlp_ratio=4.)

        # 全結合層による分類ヘッド (以前のコードと同じものを維持)
        # PoolFormerブロックの出力後の特徴マップをグローバルプーリングしてフラット化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        # R(2+1)Dブロックとそれに続くプーリング層を適用
        x = self.r2plus1d_block1(x)
        x = self.pool1(x)
        # print("Shape after R2+1D block1/pool1:", x.shape) # デバッグ用

        x = self.r2plus1d_block2(x)
        x = self.pool2(x)
        # print("Shape after R2+1D block2/pool2:", x.shape) # デバッグ用

        x = self.r2plus1d_block3(x)
        x = self.pool3(x)
        # print("Shape before PoolFormer block:", x.shape) # デバッグ用

        # PoolFormerブロックを適用
        x = self.poolformer_block(x)
        # print("Shape after PoolFormer block:", x.shape) # デバッグ用

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class I3DWithMobileAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, width_multiplier=0.5):
        super(I3DWithMobileAttention, self).__init__()

        # 各層のチャンネル数を width_multiplier で調整
        base_ch = int(64 * width_multiplier)

        # ブロック1: 入力 -> base_ch
        self.conv1 = nn.Conv3d(in_channels, base_ch, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ブロック2: Inceptionモジュール1 (チャンネル数を削減)
        self.inception1 = InceptionBlock3D(base_ch,
                                          branch1x1_out=int(64*width_multiplier),
                                          branch3x3_reduce=int(96*width_multiplier),
                                          branch3x3_out=int(128*width_multiplier),
                                          branch5x5_reduce=int(16*width_multiplier),
                                          branch5x5_out=int(32*width_multiplier),
                                          branch_pool_out=int(32*width_multiplier))

        # Inception1の出力チャンネル数を計算
        inception1_out = int((64+128+32+32)*width_multiplier)

        # ブロック3: Inceptionモジュール2
        self.inception2 = InceptionBlock3D(inception1_out,
                                          branch1x1_out=int(128*width_multiplier),
                                          branch3x3_reduce=int(128*width_multiplier),
                                          branch3x3_out=int(192*width_multiplier),
                                          branch5x5_reduce=int(32*width_multiplier),
                                          branch5x5_out=int(96*width_multiplier),
                                          branch_pool_out=int(64*width_multiplier))

        # Inception2の出力チャンネル数を計算
        inception2_out = int((128+192+96+64)*width_multiplier)

        # 残りの部分もチャンネル数を調整
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.attention_module = MobileAttention3D(in_channels=inception2_out, num_heads=8, key_dim=64, value_dim=64)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(inception2_out, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)

        # 基本畳み込みブロック
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        # Inceptionブロック
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)

        # アテンションモジュールを適用
        x = self.attention_module(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class I3DWithPoolFormer(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, width_multiplier=0.5):
        super(I3DWithPoolFormer, self).__init__()

        # 各層のチャンネル数を width_multiplier で調整
        base_ch = int(64 * width_multiplier)

        # ブロック1: 入力 -> base_ch
        self.conv1 = nn.Conv3d(in_channels, base_ch, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ブロック2: Inceptionモジュール1 (チャンネル数を削減)
        self.inception1 = InceptionBlock3D(base_ch,
                                          branch1x1_out=int(64*width_multiplier),
                                          branch3x3_reduce=int(96*width_multiplier),
                                          branch3x3_out=int(128*width_multiplier),
                                          branch5x5_reduce=int(16*width_multiplier),
                                          branch5x5_out=int(32*width_multiplier),
                                          branch_pool_out=int(32*width_multiplier))

        # Inception1の出力チャンネル数を計算
        inception1_out = int((64+128+32+32)*width_multiplier)

        # ブロック3: Inceptionモジュール2
        self.inception2 = InceptionBlock3D(inception1_out,
                                          branch1x1_out=int(128*width_multiplier),
                                          branch3x3_reduce=int(128*width_multiplier),
                                          branch3x3_out=int(192*width_multiplier),
                                          branch5x5_reduce=int(32*width_multiplier),
                                          branch5x5_out=int(96*width_multiplier),
                                          branch_pool_out=int(64*width_multiplier))

        # Inception2の出力チャンネル数を計算
        inception2_out = int((128+192+96+64)*width_multiplier)

        # 残りの部分もチャンネル数を調整
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        # MobileAttention3D を PoolFormer3DBlock に置き換え
        # conv3/pool3 の出力チャネル数128を入力とする
        self.poolformer_block = PoolFormer3DBlock(dim=inception2_out, pool_size=(1, 3, 3), mlp_ratio=4.)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(inception2_out, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)

        # 基本畳み込みブロック
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        # Inceptionブロック
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)

        # アテンションモジュールを適用
        x = self.poolformer_block(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# 基本的な3D-CNNモデル（アテンションなし）
class CNN3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(CNN3D, self).__init__()

        # 3D CNN層 (特徴量を抽出)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 全結合層による分類ヘッド
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# R(2+1)Dモデル（アテンションなし）
class R2plus1d(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(R2plus1d, self).__init__()

        # R(2+1)D分解された畳み込みブロック
        # ブロック1: 3->32 ch
        self.r2plus1d_block1 = R2plus1dBlock(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # ブロック2: 32->64 ch
        self.r2plus1d_block2 = R2plus1dBlock(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ブロック3: 64->128 ch
        self.r2plus1d_block3 = R2plus1dBlock(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), mid_channels_ratio=0.5)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 全結合層による分類ヘッド
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # R(2+1)Dブロックとプーリングを順に適用
        x = self.pool1(self.r2plus1d_block1(x))
        x = self.pool2(self.r2plus1d_block2(x))
        x = self.pool3(self.r2plus1d_block3(x))

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# I3Dモデル（アテンションなし）
class I3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, width_multiplier=0.5):
        super(I3D, self).__init__()

        # 各層のチャンネル数を width_multiplier で調整
        base_ch = int(64 * width_multiplier)

        # ブロック1: 入力 -> base_ch
        self.conv1 = nn.Conv3d(in_channels, base_ch, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ブロック2: Inceptionモジュール1
        self.inception1 = InceptionBlock3D(base_ch,
                                          branch1x1_out=int(64*width_multiplier),
                                          branch3x3_reduce=int(96*width_multiplier),
                                          branch3x3_out=int(128*width_multiplier),
                                          branch5x5_reduce=int(16*width_multiplier),
                                          branch5x5_out=int(32*width_multiplier),
                                          branch_pool_out=int(32*width_multiplier))

        # Inception1の出力チャンネル数を計算
        inception1_out = int((64+128+32+32)*width_multiplier)

        # ブロック3: Inceptionモジュール2
        self.inception2 = InceptionBlock3D(inception1_out,
                                          branch1x1_out=int(128*width_multiplier),
                                          branch3x3_reduce=int(128*width_multiplier),
                                          branch3x3_out=int(192*width_multiplier),
                                          branch5x5_reduce=int(32*width_multiplier),
                                          branch5x5_out=int(96*width_multiplier),
                                          branch_pool_out=int(64*width_multiplier))

        # Inception2の出力チャンネル数を計算
        inception2_out = int((128+192+96+64)*width_multiplier)

        # 空間サイズを調整するプーリング
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        # 全結合層による分類ヘッド
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(inception2_out, num_classes)

    def forward(self, x):
        # 基本畳み込みブロック
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))

        # Inceptionブロック
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)

        # 分類ヘッド
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# モデルのインスタンス化と確認
# 入力データの例: バッチサイズ 2, チャンネル 3 (RGB), フレーム数 5, 高さ 64, 幅 64
# input_tensor = torch.randn(2, 3, 5, 64, 64)
# model = BasicCNN3DWithMobileAttention(in_channels=3, num_classes=7)
#
# output = model(input_tensor)
# print("モデルの出力形状:", output.shape)

# モデルの構造を表示 (オプション)
# print(model)