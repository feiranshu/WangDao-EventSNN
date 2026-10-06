"""核对 EV-SpSegNet 论文值与官方仓库当前代码的参数量。

本脚本不下载代码、不执行稀疏卷积，只按官方 commit 33e2aff 中
``model/evspsegnet.py``、``model/basemodel.py`` 和 width=12 配置展开
所有可训练张量。SparseConv/SubMConv/InverseConv 的权重元素数均为
kernel_volume * in_channels * out_channels。
"""

from __future__ import annotations


WIDTH = 12
INPUT_CHANNELS = 4
OFFICIAL_COMMIT = "33e2affb51a204676536e119803ab8494e2eb0a9"
PAPER_REPORTED_PARAMETERS = 4_000_000  # 论文 Table 2 的 4.0M（舍入值）


def batch_norm(channels: int) -> int:
    return 2 * channels  # trainable weight + bias


def sparse_conv(in_channels: int, out_channels: int, kernel_volume: int, bias: bool = False) -> int:
    return in_channels * out_channels * kernel_volume + (out_channels if bias else 0)


def gd_block(in_channels: int, out_channels: int, add_channels: int = 16) -> int:
    channels = out_channels + add_channels
    group_channels = channels // 4
    shortcut = sparse_conv(in_channels, out_channels, 1) + batch_norm(out_channels)
    pointwise = sparse_conv(in_channels, channels, 1) + batch_norm(channels)
    # GDConv does not pass bias=False, and spconv.SubMConv3d defaults to bias=True.
    grouped_dilated = 4 * sparse_conv(group_channels, group_channels, 27, bias=True) + batch_norm(channels)
    squeeze_excitation = channels * (channels // 2) + (channels // 2) * channels
    projection = sparse_conv(channels, out_channels, 1) + batch_norm(out_channels)
    outer_norm = batch_norm(out_channels)
    return shortcut + pointwise + grouped_dilated + squeeze_excitation + projection + outer_norm


def regular_block(in_channels: int, out_channels: int) -> int:
    return sparse_conv(in_channels, out_channels, 27) + batch_norm(out_channels)


def residual_block(channels: int) -> int:
    return 2 * sparse_conv(channels, channels, 27) + 2 * batch_norm(channels)


def patch_attention(channels: int, inverse_layers: int) -> int:
    inverse_convs = inverse_layers * sparse_conv(channels, channels, 16)
    final_conv = sparse_conv(channels, channels, 1)
    multihead_attention = 4 * channels * channels + 4 * channels
    return inverse_convs + final_conv + multihead_attention


def current_code_parameter_count(width: int = WIDTH) -> int:
    parts = [
        sparse_conv(INPUT_CHANNELS, width, 27) + batch_norm(width),
        gd_block(width, width),
        regular_block(width, 2 * width),
        gd_block(2 * width, 2 * width),
        patch_attention(2 * width, 4),
        regular_block(2 * width, 4 * width),
        gd_block(4 * width, 4 * width, 8),
        patch_attention(4 * width, 3),
        regular_block(4 * width, 4 * width),
        gd_block(4 * width, 4 * width, 0),
        patch_attention(4 * width, 2),
        residual_block(4 * width),
        gd_block(8 * width, 4 * width),
        regular_block(4 * width, 4 * width),
        residual_block(4 * width),
        gd_block(8 * width, 4 * width),
        regular_block(4 * width, 2 * width),
        residual_block(2 * width),
        gd_block(4 * width, 2 * width),
        regular_block(2 * width, width),
        residual_block(width),
        gd_block(2 * width, width),
        gd_block(width, width),
        width + 1,  # final Linear weight + bias
    ]
    return sum(parts)


if __name__ == "__main__":
    current = current_code_parameter_count()
    print(f"paper reported: {PAPER_REPORTED_PARAMETERS:,} parameters (rounded 4.0M)")
    print(f"official commit {OFFICIAL_COMMIT[:7]} width=12: {current:,} trainable parameters")
    print(f"current-code FP32 parameter payload: {current * 4:,} bytes")
