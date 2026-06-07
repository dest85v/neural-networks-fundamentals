import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        ## YOUR CODE HERE
        # Define network modules in the constructor
        super().__init__()
        if channels is None:
            channels = [784]
        layer_sizes = [channels[0]] + list(channels[1:]) + [n_classes]
        backbone_layers = []
        for i in range(len(layer_sizes) - 2):
            backbone_layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            backbone_layers.append(activation())
        self.backbone = torch.nn.Sequential(*backbone_layers)
        self.classifier = torch.nn.Linear(channels[-1], n_classes)
        
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        signal = self.backbone(signal)
        signal = self.classifier(signal)
        
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        
        # Put the result into the batch
        batch['signals'] = {'output': signal}
        
        # Perform postprocessing after we get the output
        self.postprocessing(batch)
        
        return batch['signals']['output']
    
    def postprocessing(self, batch):
        
        # Take network's output from the batch
        signal = batch['signals']['output']
        
        ## YOUR CODE HERE
        signal = torch.argmax(signal, dim=1)
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
