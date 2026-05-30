import torch

'''
    RNN seq2seq model
    x.shape = (batch, seq_length, num_inputs)
    y.shape = (batch, seq_length, num_outputs)
'''
class RnnModelDetector(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(RnnModelDetector, self).__init__()

        self.rnn    = torch.nn.GRU(num_inputs, num_hidden, batch_first=True)
        self.fc     = torch.nn.Linear(num_hidden, num_outputs)  
        
        torch.nn.init.orthogonal_(self.rnn.weight_ih_l0, gain=0.5)
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0, gain=0.5)

        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

        torch.nn.init.orthogonal_(self.fc.weight, gain=0.01)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        rnn_output, hidden = self.rnn(x)  

        output = self.fc(rnn_output)
        return output
  


class RnnModelClassifier(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super(RnnModelClassifier, self).__init__()

        kernel_size = 7 

        self.conv0  = torch.nn.Conv1d(num_inputs, num_hidden, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//2)
        self.act0   = torch.nn.ReLU()
        self.conv1  = torch.nn.Conv1d(num_hidden, num_hidden, kernel_size=kernel_size, stride=kernel_size//2, padding=kernel_size//2)
        self.act1   = torch.nn.ReLU()   

        self.rnn    = torch.nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.fc     = torch.nn.Linear(num_hidden, num_outputs)  

        torch.nn.init.orthogonal_(self.conv0.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv1.bias)
        
        torch.nn.init.orthogonal_(self.rnn.weight_ih_l0, gain=0.5)
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0, gain=0.5)

        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

        torch.nn.init.orthogonal_(self.fc.weight, gain=0.01)
        torch.nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = torch.transpose(x, 2, 1)


        rnn_output, hidden = self.rnn(x)  

        last_output = hidden.squeeze(0)

        output = self.fc(last_output)
        return output
    



class CNNModelClassifier(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CNNModelClassifier, self).__init__()

        kernel_size = 7     

        self.conv0  = torch.nn.Conv1d(num_inputs, 16, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.act0   = torch.nn.ReLU()
        self.conv1  = torch.nn.Conv1d(16, 32, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.act1   = torch.nn.ReLU()   
        self.conv2  = torch.nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.act2   = torch.nn.ReLU()   
        self.conv3  = torch.nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.act3   = torch.nn.ReLU()   

        self.lin0   = torch.nn.Linear(64*32, num_outputs)
       

        torch.nn.init.orthogonal_(self.conv0.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.orthogonal_(self.conv2.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.orthogonal_(self.conv3.weight, gain=0.5)
        torch.nn.init.zeros_(self.conv3.bias)

        torch.nn.init.orthogonal_(self.lin0.weight, gain=0.01)
        torch.nn.init.zeros_(self.lin0.bias)
        

    
    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)

        x = x.flatten(1)

        output = self.lin0(x)
        return output
    

if __name__ == "__main__":
    x = torch.randn((10, 1024, 3))

    '''
    detection = RnnModelDetector(3, 1, 64)
    y = detection(x)
    print(y.shape)

    classifier = RnnModelClassifier(3, 5, 64)
    print(classifier)
    y = classifier(x)
    print(y.shape)
    '''

    classifier = CNNModelClassifier(3, 5)
    print(classifier)
    y = classifier(x)
    print(y.shape)



