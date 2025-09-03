import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # 定义 Sobel 核，用于提取水平和垂直方向的边缘
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # 水平方向
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  # 垂直方向

        # 转换为 PyTorch 张量
        self.sobel_x = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )
        self.sobel_y = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        # 初始化 Sobel 滤波器的权重
        self.sobel_x.weight.data = torch.tensor(sobel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y.weight.data = torch.tensor(sobel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 锁定权重，防止训练中更新
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # 如果输入有多个通道，逐通道应用 Sobel 滤波器
        n, c, h, w = x.size()
        edges = []
        for i in range(c):
            single_channel = x[:, i:i+1, :, :]  # 提取单个通道
            edge_x = self.sobel_x(single_channel)
            edge_y = self.sobel_y(single_channel)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # 计算梯度幅值
            edges.append(edge)

        # 将每个通道的结果拼接在一起
        edges = torch.cat(edges, dim=1)
        return edges


class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()
        self.r = r
        self.eps = eps
        self.sobel_filter = SobelFilter()  # 使用 Sobel 滤波器
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        # 获取输入尺寸
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # 输入校验
        assert n_lrx == n_lry == n_hrx, "Input batch sizes must match."
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry), "Channel dimensions must align."
        assert h_lrx == h_lry and w_lrx == w_lry, "Low-resolution dimensions must match."
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1, "Input dimensions must be larger than the filter size."

        # 使用 Sobel 滤波器提取边缘信息
        N = self.sobel_filter(torch.ones((n_lrx, c_lrx, h_lrx, w_lrx), device=lr_x.device, dtype=lr_x.dtype))

        # 处理 l_a 和权重归一化
        l_a = torch.abs(l_a) + self.epss
        t_all = torch.sum(l_a, dim=(2, 3), keepdim=True)
        l_t = l_a / t_all

        # 预计算部分
        mean_a = self.sobel_filter(l_a) / N
        mean_ax = self.sobel_filter(l_a * lr_x) / N
        mean_ay = self.sobel_filter(l_a * lr_y) / N
        mean_a2x2 = self.sobel_filter(l_a * l_a * lr_x * lr_x) / N
        mean_a2xy = self.sobel_filter(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.sobel_filter(l_t * l_a * lr_x) / N

        # 线性模型 A 和 b 的计算
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a + self.eps)

        # 平滑 A 和 b
        A = self.sobel_filter(A) / N
        b = self.sobel_filter(b) / N

        # 高分辨率插值
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        # 返回结果
        return (mean_A * hr_x + mean_b).float()

if __name__ == "__main__":
    # 定义示例参数
    r = 4
    eps = 1e-8
    b, c, h, w = 2, 512, 36, 36  # Batch size, Channels, Height, Width

    # 创建随机输入
    lr_x = torch.rand((b, c, h, w), dtype=torch.float32)
    lr_y = torch.rand((b, c, h, w), dtype=torch.float32)
    hr_x = torch.rand((b, c, h * 2, w * 2), dtype=torch.float32)
    l_a = torch.rand((b, c, h, w), dtype=torch.float32)

    # 初始化模型
    model = FastGuidedFilter_attention(r, eps)

    # 运行模型
    output = model(lr_x, lr_y, hr_x, l_a)

    # 输出结果
    print(f"Output shape: {output.shape}")