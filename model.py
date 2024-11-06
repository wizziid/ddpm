import torch
import torch.nn as nn
import torch.nn.functional as F


# THE MODEL
# toggle batch norm or group norm and conditioning
norm = "g"
conditioning = False


class SelfAttention(nn.Module):

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


# middle unet block
class MiddleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MiddleBlock, self).__init__()
        out_channels = in_channels*2
        # 4 convs?
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, in_channels, 3, padding=1)

        self.norm1 = nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)
        self.norm4 = nn.BatchNorm2d(in_channels) if norm=="b" else nn.GroupNorm(32, in_channels)

        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
      x = self.down_sample(x)

      x = F.silu(self.norm1(self.conv1(x)))
      x = F.silu(self.norm2(self.conv2(x)))
      x = F.silu(self.norm3(self.conv3(x)))
      x = F.silu(self.norm4(self.conv4(x)))

      return x

# down unet block
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_size, down = True):
        super(DownBlock, self).__init__()

        self.down = down
        self.down_sample = nn.MaxPool2d(2)

        self.time_network = nn.Linear(time_embedding_size, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)


        self.norm1 = nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)

    def forward(self, x, time_embedding):
        if self.down:
          # this occurs every layer except the first
          x = self.down_sample(x)


        y = F.silu(self.norm1(self.conv1(x)))
        y = F.silu(self.norm2(self.conv2(y)))
        time_embedding = self.time_network(time_embedding)[(..., ) + (None, ) * 2]

        return y + time_embedding


# up unet block
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_size, out = False):
        super(UpBlock, self).__init__()

        self.time_network = nn.Linear(time_embedding_size, out_channels)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.time_network = nn.Linear(time_embedding_size, out_channels)

        self.conv1 = nn.Conv2d(in_channels,in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1)

        # if out == False:
        self.norm1 = nn.BatchNorm2d(in_channels//2) if norm=="b" else nn.GroupNorm(32, in_channels//2)
        self.norm2 =  nn.BatchNorm2d(out_channels) if norm=="b" else nn.GroupNorm(32, out_channels)



    def forward(self, x, bridged_x, time_embedding):
        x =self.up(x)
        x = torch.concat([x, bridged_x], dim=1)
        y = F.silu(self.norm1(self.conv1(x)))
        y = F.silu(self.norm2(self.conv2(y)))
        time_embedding = self.time_network(time_embedding)[(..., ) + (None, ) * 2]

        return y + time_embedding



# basic DDPM using markov chain like paper

class DiffusionSimple(nn.Module):
    def __init__(self, device, resolution):
        super(DiffusionSimple, self).__init__()
        self.resolution = resolution
        self.device = device
        self.beta_0 = 1e-4
        self.beta_T = 0.01
        self.T = 200
        self.time_embedding_size = 256


        # may need more blocks?
        self.down_channels = [64, 128, 256, 512]
        self.up_channels =  [512, 256, 128, 64]

        # betas and alphas
        self.betas = torch.linspace(self.beta_0, self.beta_T, self.T).to(device)
        self.alphas = 1.-self.betas
        self.alphas_cumprod = torch.cumprod((self.alphas), axis=0)


        # https://paperswithcode.com/paper/denoising-diffusion-probabilistic-models
        # paper uses residual blocks so this implementation is based on what they
        # used in appendix B (With less overall parameters). Try without attention
        # to begin with


        self.time_mlp = nn.Linear(self.time_embedding_size, self.time_embedding_size)

        # Channels: 3 -> 64 channels. Pixels: 64x64 to 64x64
        self.down_block_one = DownBlock(
            in_channels = 3,
            out_channels = self.down_channels[0],
            time_embedding_size = self.time_embedding_size,
            down = False
        )

        # Channels: 64 -> 128 channels. Pixels: 64x64 to 32x32
        self.down_block_two = DownBlock(
            in_channels = self.down_channels[0],
            out_channels = self.down_channels[1],
            time_embedding_size = self.time_embedding_size
        )

        # Channels: 128 -> 256 channels. Pixels: 32x32 to 16x16 -- This is where they put global attention in paper?
        self.down_block_three = DownBlock(
            in_channels = self.down_channels[1],
            out_channels = self.down_channels[2],
            time_embedding_size = self.time_embedding_size
        )

        # Channels: 256 -> 512 channels. Pixels: 16x16 to 8x8
        self.down_block_four = DownBlock(
            in_channels = self.down_channels[2],
            out_channels = self.down_channels[3],
            time_embedding_size = self.time_embedding_size
        )

        # Channels: 512 -> 1024 -> 512. Pixels: 8x8 -> 4x4
        self.middle_block = MiddleBlock(self.down_channels[3])


        # Channels: 1024 -> 256. Pixels: 4x4 -> 8x8
        self.up_block_one = UpBlock(
            in_channels = self.up_channels[0]*2,
            out_channels = self.up_channels[1],
            time_embedding_size = self.time_embedding_size
        )

        # Channels: 512 -> 128. Pixels: 8x8 -> 16x16
        self.up_block_two = UpBlock(
            in_channels = self.up_channels[1]*2,
            out_channels = self.up_channels[2],
            time_embedding_size = self.time_embedding_size
        )

        # Channels: 256 -> 64. Pixels: 16x16 -> 32x32
        self.up_block_three = UpBlock(
            in_channels = self.up_channels[2]*2,
            out_channels = self.up_channels[3],
            time_embedding_size = self.time_embedding_size
        )



        # Channels: 128 -> 64. Pixels: 32x32 -> 64x64
        self.up_block_four = UpBlock(
            in_channels = self.up_channels[3]*2,
            out_channels = self.up_channels[3],
            time_embedding_size = self.time_embedding_size,
            out=True
        )


        self.last_conv = nn.Conv2d(self.up_channels[3], 3, kernel_size=1)


        # these come after each non input/output resblock

        """
        self.att1 = SelfAttention(self.down_channels[1],  32)
        self.att2 = SelfAttention(self.down_channels[2], 16)
        self.att3 = SelfAttention(self.down_channels[3], 8)
        self.att4 = SelfAttention(self.up_channels[1], 8)
        self.att5 = SelfAttention(self.up_channels[2], 16)
        self.att6 = SelfAttention(self.up_channels[3], 32)
        """




    def network(self, y_t, t):
        """
        Network that predicts noise added to y_t-1 to reach y_t
        """

        # combined = torch.cat([y_t, t], dim=1).to(device)
        
        time_embedding = F.silu(self.time_mlp(self.time_embedding(t)))
        y_t_1 = self.down_block_one(y_t, time_embedding)
        y_t_2 = self.down_block_two(y_t_1, time_embedding)
        #y_t_2 = self.att1(y_t_2)
        y_t_3 = self.down_block_three(y_t_2, time_embedding)
        #y_t_3 = self.att2(y_t_3)
        y_t_4 = self.down_block_four(y_t_3, time_embedding)
        #y_t_4 = self.att3(y_t_4)

        y_t_5 = self.middle_block(y_t_4)

        y_t_6 = self.up_block_one(y_t_5, y_t_4, time_embedding)
        #y_t_6 = self.att4(y_t_6)
        y_t_7 = self.up_block_two(y_t_6, y_t_3, time_embedding)
        #y_t_7 = self.att5(y_t_7)
        y_t_8 = self.up_block_three(y_t_7, y_t_2, time_embedding)
        # y_t_8 = self.att6(y_t_8)
        y_t_9 = self.up_block_four(y_t_8, y_t_1, time_embedding)

        return self.last_conv(y_t_9)


    @torch.no_grad()
    def sample_img(self, n):

        self.eval()
        with torch.no_grad():
            steps = []
            y_t = torch.randn(n, 3, self.resolution, self.resolution).to(self.device )
            for i in reversed(range(1, self.T)):
                
                t = (torch.ones(1).to(self.device ) * i).long()
                predicted_noise = self.network(y_t, t)
                alphas = self.alphas[t][:, None, None, None]
                alphas_cumprod = self.alphas_cumprod[t][:, None, None, None]
                betas = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(y_t).to(self.device )
                else:
                    noise = torch.zeros_like(y_t).to(self.device )
                y_t = 1 / torch.sqrt(alphas) * (y_t - ((1 - alphas) / (torch.sqrt(1 - alphas_cumprod))) * predicted_noise) + torch.sqrt(betas) * noise

                if i <= 200 and i%50 == 0:
                    # return last 200 steps for plotting
                    steps.append(y_t.squeeze())
        self.train()

        return y_t, steps

    # smaple_t may not be best name for this method
    def sample_t(self, y_0, t):
        """
        Take an image and noise it using closed form sampling, also return the noise
        added for loss function
        """
        noise = torch.randn_like(y_0).to(self.device)
        mean = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None] * y_0
        std = torch.sqrt(1-self.alphas_cumprod[t])[:, None, None, None]  * noise
        y_t = mean+std
        return y_t, noise
    
    def time_embedding(self, t):
        i = torch.arange(0, self.time_embedding_size // 2).float().to(self.device)
        time_steps = t.view(-1, 1)

        sin_emb = (time_steps / (10000 ** ((2 * i) / self.time_embedding_size))).sin()
        cos_emb = (time_steps / (10000 ** ((2 * i) / self.time_embedding_size))).cos()
        embeddings = torch.cat((sin_emb, cos_emb), dim=1)

        return embeddings

    
