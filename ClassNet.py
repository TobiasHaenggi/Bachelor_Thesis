# Neural Network neural networking

# NEURAL NETWORK

class ClassNet(nn.Module):

  def __init__(self, fc_l_1=200, fc_l_2=150):

     super(ClassNet, self).__init__()

     # Layer 1
     self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
     # Layer 2
     self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
     # FC layers
     self.linear1 = nn.Linear(16 * 5 * 5, fc_l_1, bias=True) 
     self.linear2 = nn.Linear(fc_l_1, fc_l_2, bias=True) 
     self.linear3 = nn.Linear(fc_l_2, 100) 
        
     # add a softmax to the last layer
     self.logsoftmax = nn.LogSoftmax(dim=1) 
     
     # activation function
     self.relu = nn.ReLU(inplace=True)

  def forward(self, images): # ((NTS: The actual learning part))

     # conv layers
     x = self.pool1(self.relu(self.conv1(images)))
     x = self.pool2(self.relu(self.conv2(x)))

     # reshape image pixels
     x = x.view(-1, 16 * 5 * 5)
 
     # FC Layers
     x = self.relu(self.linear1(x))
     x = self.relu(self.linear2(x))
        
     # log softmax
     x = self.logsoftmax(self.linear3(x))

     return x
