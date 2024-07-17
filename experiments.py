# under development
class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_blocks: int = 8, 
        sparsity_threshold: float = 0.01, 
        hard_thresholding_fraction: float = 1., 
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size_factor = hidden_size_factor

        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        self.w1 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size, self.block_size * hidden_size_factor)
        )
        self.b1 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size * hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(
            data=self.scale * torch.randn(2, num_blocks, self.block_size)
        )

    def forward(
        self, 
        input: torch.Tensor, 
    ):
        
        assert input.ndim == 4
        batch_size: int = input.shape[0]
        x_res: int = input.shape[1]
        y_res: int = input.shape[2]
        u_dim: int = input.shape[3]

        bias: torch.Tensor = x
        dtype = x.dtype

        out: torch.Tensor = torch.fft.rfft2(input, dim=(1, 2), norm="ortho")
        out: torch.Tensor = out.reshape(batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size)

        out1_real: torch.Tensor = torch.zeros(
            (batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor), 
            device=x.device,
        )
        out1_imag: torch.Tensor = torch.zeros(
            (batch_size, out.shape[1], out.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor), 
            device=x.device,
        )
        out2_real: torch.Tensor = torch.zeros(out.shape, device=x.device)
        out2_imag: torch.Tensor = torch.zeros(out.shape, device=x.device)

        total_modes: int = (x_res * y_res) // 2 + 1
        kept_modes: int = int(total_modes * self.hard_thresholding_fraction)

        

        out1_real[:, :, :kept_modes, :, :] = F.relu(
            torch.einsum('Bxmbi,bio->Bxmbo', out[:, :, :kept_modes, :, :].real, self.w1[0])
            - torch.einsum('...bi,bio->...bo', out[:, :, :kept_modes, :, :].imag, self.w1[1])
            + self.b1[0]
        )

        out1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias


