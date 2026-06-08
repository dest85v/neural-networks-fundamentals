import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE
        #return grad_output, None
        return grad_output * (-ctx.strength), None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        ## YOUR CODE HERE
        super().__init__()
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge
        self.activation = activation

        input_dim, hidden_dim, output_dim = channels
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Tanh(),
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            activation(),
            torch.nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = torch.nn.Linear(input_dim, 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
#            generated = batch['data'].get('noise')
#            if generated is None:
#                generated = torch.empty(0)
#            batch['signals'] = {
#                'generated': generated,
#                'fake_scores': torch.zeros(generated.shape[0], device=generated.device),
#                'fake_logits': torch.zeros(generated.shape[0], device=generated.device),
#            }
#            batch['postprocessed'] = {
#                'fake_score': torch.zeros(generated.shape[0], device=generated.device),
#                'fake_probability': torch.zeros(generated.shape[0], device=generated.device),
#            }
            batch['signals'] = {}
            batch['postprocessed'] = {}

        noise = batch['data']['noise']
        real_image = batch['data'].get('real', batch['data'].get('image', None))

        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)

        if real_image is not None:
            real_flat = real_image.reshape(real_image.shape[0], -1)
            discriminator_input = torch.cat([reversed_generated, real_flat], dim=0)
        else:
            discriminator_input = reversed_generated

        discriminator_logits = self.discriminate(discriminator_input)

        n_gen = generated.shape[0]
        if real_image is not None:
            fake_logits = discriminator_logits[:n_gen]
            real_logits = discriminator_logits[n_gen:]
        else:
            fake_logits = discriminator_logits
            real_logits = torch.empty(0, device=discriminator_logits.device)

        batch['signals']['generated'] = generated
        batch['signals']['discriminator_logits'] = discriminator_logits
        batch['signals']['fake_logits'] = fake_logits
        batch['signals']['real_logits'] = real_logits
        batch['signals']['discriminator_scores'] = torch.sigmoid(discriminator_logits)
        batch['signals']['fake_scores'] = torch.sigmoid(fake_logits)
        batch['signals']['real_scores'] = torch.sigmoid(real_logits)

        batch['postprocessed']['generated'] = generated
        batch['postprocessed']['discriminator_score'] = batch['signals']['discriminator_scores']
        batch['postprocessed']['discriminator_probability'] = torch.sigmoid(batch['signals']['discriminator_scores'])
        batch['postprocessed']['fake_score'] = batch['signals']['fake_scores']
        batch['postprocessed']['fake_probability'] = torch.sigmoid(batch['signals']['fake_scores'])
        batch['postprocessed']['real_score'] = batch['signals']['real_scores'] if real_logits.numel() > 0 else torch.tensor([], device=discriminator_logits.device)
        batch['postprocessed']['real_probability'] = torch.sigmoid(batch['signals']['real_scores']) if real_logits.numel() > 0 else torch.tensor([], device=discriminator_logits.device)
        return batch
