import torch
import torch.nn as nn
import torch.nn.functional as F

# R(2+1)D分解された単一の畳み込みブロック
class R2plus1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1,
                 mid_channels_ratio=1.0):  # mid_channels_ratio パラメータを追加
        super(R2plus1dBlock, self).__init__()

        # カーネルサイズ、パディング、ストライドはタプルで指定 (時間/奥行き, 高さ, 幅)
        # 例: kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)

        # strideをタプルに変換（intで渡された場合に対応）
        stride_tuple = stride if isinstance(stride, tuple) else (stride, stride, stride)

        # 空間畳み込み (時間/奥行き方向のカーネルサイズは1)
        # 入力チャネル -> 中間チャネル
        # ここで中間チャネル数をout_channels * mid_channels_ratio で計算
        mid_channels = int(out_channels * mid_channels_ratio)
        # 中間チャンネル数が0にならないように最低値を設けるなどの考慮も必要かもしれません
        mid_channels = max(1, mid_channels)  # 少なくとも1チャンネルは確保

        # 空間畳み込みのパディングは、時間/奥行き方向は0、空間方向は元のパディングを使用
        spatial_padding = (0, padding[1], padding[2])
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        # 空間畳み込みのストライドは、時間方向は1、空間方向は元のストライドを使用
        spatial_stride = (1, stride_tuple[1], stride_tuple[2])

        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,  # 修正: 中間チャンネルを使用
            kernel_size=spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            bias=False  # BNを使用するためBiasはFalseが一般的
        )
        self.spatial_bn = nn.BatchNorm3d(mid_channels)  # 修正: 中間チャンネルを使用
        self.relu = nn.ReLU(inplace=True)

        # 時間/奥行き畳み込み (空間方向のカーネルサイズは1)
        # 中間チャネル -> 出力チャネル
        # 時間/奥行き畳み込みのパディングは、空間方向は0、時間/奥行き方向は元のパディングを使用
        temporal_padding = (padding[0], 0, 0)
        temporal_kernel_size = (kernel_size[0], 1, 1)
        # 時間畳み込みのストライドは、時間方向は元のストライド、空間ストライドは1
        temporal_stride = (stride_tuple[0], 1, 1)

        self.temporal_conv = nn.Conv3d(
            mid_channels,  # 修正: 中間チャンネルを入力として使用
            out_channels,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=False  # BNを使用するためBiasはFalseが一般的
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)
        # 最後のReLUは残す設計と残さない設計がありますが、ここではブロック内に含めます
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # 空間畳み込みとそれに続くBN, ReLU
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)

        # 時間/奥行き畳み込みとそれに続くBN, ReLU
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.relu2(x)  # 最後のReLU

        return x


# MobileNetV4のMobileAttentionブロック構造を参考にした3Dアテンションモジュール
class MobileAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=8, key_dim=64, value_dim=64, attn_drop=0.0, proj_drop=0.0):
        super(MobileAttention3D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        # Mobile-MQAのKey/Value共有の概念を取り入れ、KとVはQueryとは異なる射影を使用
        self.head_dim = value_dim  # 出力はvalue_dimの合計となるため

        self.scale = self.head_dim ** -0.5

        # Q, K, Vの射影層
        # 3D畳み込みを使用し、空間次元と時間次元を扱えるようにする
        # カーネルサイズ1x1x1でチャネル数を調整するようなイメージ
        self.q = nn.Conv3d(in_channels, num_heads * key_dim, kernel_size=1)
        # KeyとValueは共有のため、Key headとValue headは1つとして扱う (Mobile-MQAの簡易的な3D化)
        self.kv = nn.Conv3d(in_channels, key_dim + value_dim, kernel_size=1)

        # ドロップアウト
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(num_heads * value_dim, in_channels, kernel_size=1)  # 出力射影
        self.proj_drop = nn.Dropout(proj_drop)

        # LayerScale (MobileNetV4のblocks.pyにあるLayerScale2dを3Dに簡易化)
        # 論文の実装ではLayerScale2dがConvNextやExtraDWの後に来ているが、Attentionの後にも適用を想定
        self.layer_scale = nn.Parameter(torch.ones(in_channels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 1e-5)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        B, C, D, H, W = x.shape

        # Q, K, Vの計算
        q = self.q(x).reshape(B, self.num_heads, self.key_dim, D, H, W).permute(0, 1, 3, 4, 5,
                                                                                2)  # B, num_heads, D, H, W, key_dim
        kv = self.kv(x).reshape(B, self.key_dim + self.value_dim, D, H, W)
        k, v = torch.split(kv, [self.key_dim, self.value_dim], dim=1)
        k = k.reshape(B, 1, self.key_dim, D, H, W).permute(0, 1, 3, 4, 5, 2)  # B, 1 (shared KV head), D, H, W, key_dim
        v = v.reshape(B, 1, self.value_dim, D, H, W).permute(0, 1, 3, 4, 5,
                                                             2)  # B, 1 (shared KV head), D, H, W, value_dim

        # Mobile-MQAのKey/Value共有を模倣するため、KとVをnum_headsの次元で展開
        k = k.expand(-1, self.num_heads, -1, -1, -1, -1)  # B, num_heads, D, H, W, key_dim
        v = v.expand(-1, self.num_heads, -1, -1, -1, -1)  # B, num_heads, D, H, W, value_dim

        # Scaled Dot-Product Attention
        # QとKの転置の積: (B, num_heads, D, H, W, key_dim) @ (B, num_heads, D, H, W, key_dim) -> (B, num_heads, D, H, W, D*H*W)
        # 空間的・時間的なアテンションを計算するため、D*H*Wのflattened token dimensionに対してアテンションを計算
        q = q.flatten(3, 5)  # B, num_heads, num_tokens, key_dim
        k = k.flatten(3, 5)  # B, num_heads, num_tokens, key_dim
        v = v.flatten(3, 5)  # B, num_heads, num_tokens, value_dim

        # QK^T の計算 (Einsumに近い操作)
        # (B, num_heads, num_tokens_q, key_dim) @ (B, num_heads, key_dim, num_tokens_k) -> (B, num_heads, num_tokens_q, num_tokens_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Softmax と Dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attention Score と V の積
        # (B, num_heads, num_tokens_q, num_tokens_k) @ (B, num_heads, num_tokens_k, value_dim) -> (B, num_heads, num_tokens_q, value_dim)
        x = (attn @ v)

        # ヘッドを結合し、元の空間形状に戻す
        output_shape = (B, self.num_heads * self.head_dim, D, H, W)  # ここではアテンション前の空間サイズ D, H, W を維持
        x = x.transpose(1, 2).reshape(B, D, H, W, self.num_heads * self.head_dim).permute(0, 4, 1, 2,
                                                                                          3)  # B, num_heads * value_dim, D, H, W

        # 出力射影
        x = self.proj(x)
        x = self.proj_drop(x)

        # LayerScaleを適用
        x = x * self.layer_scale

        return x


# Position Embedding層の定義例 (学習可能)
class PositionEmbedding3D(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        # Position Embedding 用のパラメータ (D, H, W の形状に対してチャンネル数C)
        # モデルの入力サイズに応じて D, H, W を決定する必要がある
        self.position_embeddings = nn.Parameter(torch.randn(1, channels, depth, height, width))

    def forward(self, x):
        # 入力特徴マップ x に Position Embedding を加算する
        # x の形状は (batch, channels, depth, height, width)
        # position_embeddings の形状は (1, channels, depth, height, width) なので
        # バッチ次元でブロードキャストされて加算される
        return x + self.position_embeddings


# PoolFormerで使われるMlp (1x1x1 Convを使用)
class Mlp3D(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and PoolFormer
    Uses 1x1x1 convolutions for 3D data.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1)
        self.act = nn.GELU()  # PoolFormer often uses GELU
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# MetaFormer (PoolFormer) スタイルの3Dブロック
class PoolFormer3DBlock(nn.Module):
    """ 3D PoolFormer Block
    Replaces the attention mechanism with 3D pooling.
    Based on the PoolFormer: https://arxiv.org/abs/2111.11418
    """

    def __init__(self, dim, pool_size=(1, 3, 3), mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):  # Use LayerNorm as common in MetaFormers
        super().__init__()
        # LayerNorm (チャンネル次元に対して行うため、入力テンソルを転置してから適用)
        self.norm1 = norm_layer(dim)

        # Token Mixer (Pooling)
        # 平均プーリングを使用。カーネルサイズに応じたパディングで空間サイズを維持 (stride=1の場合)
        # パディング計算: pad = (kernel_size - 1) // 2
        pool_padding = tuple([(k - 1) // 2 for k in pool_size])
        self.pool = nn.AvgPool3d(pool_size, stride=1, padding=pool_padding, count_include_pad=False)
        # または depthwise Conv3d を使用することも可能:
        # self.pool = nn.Conv3d(dim, dim, pool_size, stride=1, padding=pool_padding, groups=dim)

        # LayerScale (MobileAttention3Dでも使用されていた概念、PoolFormerでも一般的)
        # (dim, 1, 1, 1)の形状にして、(B, C, D, H, W)のテンソルと要素ごとに乗算できるようにする
        self.ls1 = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 1e-5)

        # FFN (Feed-Forward Network)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ls2 = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 1e-5)

        # ドロップアウト (Stochastic Depth はここでは省略)
        self.drop_path = drop_path  # Placeholder for stochastic depth

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width) - (B, C, D, H, W)

        # LayerNormは通常、チャンネル次元に対して適用 (最後の次元にするためにpermute)
        B, C, D, H, W = x.shape
        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Mixer Branch (Pooling with Residual and LayerScale)
        mixer_out = self.pool(x_norm)
        # LayerScaleを要素wiseに乗算 (self.ls1は(C, 1, 1, 1)なのでbroadcastされる)
        x = x + mixer_out * self.ls1

        # LayerNorm before FFN
        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # FFN Branch (MLP with Residual and LayerScale)
        ffn_out = self.mlp(x_norm)
        # LayerScaleを要素wiseに乗算
        x = x + ffn_out * self.ls2

        return x


# I3DのInceptionブロック
class InceptionBlock3D(nn.Module):
    def __init__(self, in_channels, branch1x1_out, branch3x3_reduce, branch3x3_out,
                 branch5x5_reduce, branch5x5_out, branch_pool_out):
        super(InceptionBlock3D, self).__init__()

        # 1x1 畳み込み分岐
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, branch1x1_out, kernel_size=1),
            nn.BatchNorm3d(branch1x1_out),
            nn.ReLU(inplace=True)
        )

        # 1x1 畳み込み → 3x3 畳み込み分岐
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, branch3x3_reduce, kernel_size=1),
            nn.BatchNorm3d(branch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(branch3x3_reduce, branch3x3_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(branch3x3_out),
            nn.ReLU(inplace=True)
        )

        # 1x1 畳み込み → 5x5 (2つの3x3) 畳み込み分岐
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, branch5x5_reduce, kernel_size=1),
            nn.BatchNorm3d(branch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(branch5x5_reduce, branch5x5_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(branch5x5_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(branch5x5_out, branch5x5_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(branch5x5_out),
            nn.ReLU(inplace=True)
        )

        # プールから1x1畳み込み分岐
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, branch_pool_out, kernel_size=1),
            nn.BatchNorm3d(branch_pool_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 4つの分岐の出力をチャネル次元に沿って連結
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)