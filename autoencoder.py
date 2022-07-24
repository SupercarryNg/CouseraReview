import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyData(Dataset):
    def __init__(self, vec):
        self.vec = vec

    def __getitem__(self, idx):
        res = torch.from_numpy(self.vec[idx]).float()
        return res

    def __len__(self):
        return self.vec.shape[0]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        return encoded, decoded


def autoencoder_reduce(vec_ldabert):
    latent_dim = 32
    num_epochs = 100
    lr = 1e-04
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = vec_ldabert
    input_dim = X.shape[1]
    trainset = MyData(X)
    loader = DataLoader(trainset, batch_size=32, shuffle=True, drop_last=True)

    model = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    # Set progressing bar
    for epoch in range(num_epochs):
        loop = tqdm(loader, leave=False, total=len(loader))
        loss_sum = 0
        step = 0
        for inputs in loop:
            # Forward
            inputs = inputs.to(device)
            codes, decoded = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            step += 1

            loop.set_description('Epoch[{}/{}]'.format(epoch+1, num_epochs))
            loop.set_postfix(loss=loss_sum/step)

    vec, _ = model(torch.from_numpy(X).float().to(device))
    vec = vec.data.cpu().numpy()

    return vec


if __name__ == '__main__':
    model = Autoencoder(input_dim=784, latent_dim=256)
    x = torch.randn(784)
    print(model(x))
