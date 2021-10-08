import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        # lstm cell
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
    
        # linear layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
               
        
    def forward(self, features, captions):
        
        captions = captions[:, :captions.size()[1]-1]
        features = features.unsqueeze(1)
       
        captions_embed = self.embed(captions)
    
        inputs = torch.cat((features, captions_embed), dim=1)
       
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
   
        return outputs
  
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence = []
        
        while len(sentence) != (max_len+1):
            
            outputs, states = self.lstm(inputs, states)
            outputs = self.fc(outputs.squeeze(dim = 1))

            _, predicted_idx = torch.max(outputs, 1)
            
            sentence.append(predicted_idx.item())
            
            if predicted_idx == 1:
                break
            
            inputs = self.embed(predicted_idx).unsqueeze(1)
            
        return sentence