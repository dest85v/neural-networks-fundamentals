import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
#        if not hasattr(self, 'encoder'):
#            self.encoder = torch.nn.Identity()
#        if not hasattr(self, 'decoder'):
#            self.decoder = torch.nn.Identity()
        encoder_modules = []
        for i in range(len(channels) - 1):
            encoder_modules.append(torch.nn.Linear(channels[i], channels[i + 1]))
            if i < len(channels) - 2:
                encoder_modules.append(activation())
        self.encoder = torch.nn.Sequential(*encoder_modules)

        decoder_channels = list(reversed(channels))
        decoder_modules = []
        for i in range(len(decoder_channels) - 1):
            decoder_modules.append(torch.nn.Linear(decoder_channels[i], decoder_channels[i + 1]))
            if i < len(decoder_channels) - 2:
                decoder_modules.append(activation())
        self.decoder = torch.nn.Sequential(*decoder_modules)

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = signal.reshape(input_shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
#            batch['signals'] = {'reconstruction': batch['data']['image']}
            batch['signals'] = {}
        images = batch['data']['image']
        reconstruction = self.__forward_kernel(images)
        batch['signals']['reconstruction'] = reconstruction
        return batch
