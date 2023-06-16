
import torch
import torch.nn as nn

def get_z_vector(batch_size, length):
    return torch.randn(batch_size, length)

class Generator(nn.Module):
    def __init__(self, num_classes, z_dim):
        super().__init__()

        # Embedding which outputs a vector of dimension z_dim
        self.embed = nn.Embedding(num_classes, z_dim)

        # Linear combination of the latent vector z
        self.dense = nn.Linear(z_dim, 7 * 7 * 256)

        # The transposed convolutional layers are wrapped in nn.Sequential
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())

        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.trans3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh())

    def forward(self, z, label):
        # Apply embedding to the input label
        embedded_label = self.embed(label)

        # Element wise multiplication of latent vector and embedding
        x = embedded_label * z

        # Application of dense layer and transforming to 3d shape
        x = self.dense(x).view(-1, 256, 7, 7)
        x = self.trans1(x)
        x = self.trans2(x)
        x = self.trans3(x)

        return x

class AiGcMn:

    def __init__(self):
        self.device = 'cpu'  # or 'cuda'
        NUM_CLASSES = 10
        Z_DIM = 100
        path = 'generator_param3.pkl'
        self.generator = Generator(NUM_CLASSES, Z_DIM).to(self.device)
        self.generator.load_state_dict(torch.load(path))

    def generate(self, labels):
        batch_size = labels.size(0)
        z = get_z_vector(batch_size, 100).to(self.device)
        with torch.no_grad():
            self.generator.eval()
            x = self.generator(z, labels).cpu().detach()
        return x





