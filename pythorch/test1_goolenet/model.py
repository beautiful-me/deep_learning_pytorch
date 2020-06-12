import torch 
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet (nn.Module):
    def _init_(self, num_classes=1000, aux_logits=True, init_weight=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        self.conv1 = convRelu(3, 64, kernel_size=7, stride=2, padding=3)#I:224*224*3,O:112*112*64,
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        #self.LocalResponseNorm = nn.LocalResponseNorm(64, alpha=0.0001, beta=0.75, k=1.0)
         
        self.conv2 = convRelu(64,64,kernel_size=1, stride=1, padding=0)
        self.conv3 = convRelu(64,192,kernel_size=3, stride=1, padding=1)
        #self.LocalResponseNorm = nn.LocalResponseNorm(192, alpha=0.0001, beta=0.75, k=1.0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
    
        if self.Aux_logits:
            self.Aux1 = Inceptionout(512, num_classes)
            self.Aux2 = Inceptionout(528, num_classes) 
            
        self.averagepool = nn.AvgPool2d(kernel_size=7, stride=1,padding=0,ceil_mode=False)  
        self.droput = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)
        
        if init_weights: #初始化权重 ture
            self._initialize_weights()
        
    def forword(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        #if init_Aux_logits:
        if self.training and init_Aux_logits:
            Aux1 = self.Aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        
        x = self.inception4d(x)
        #if init_Aux_logits:
        if self.training and init_Aux_logits:
            Aux2 = self.Aux2(x)
            
        x = self.inception4e(x)
        x = self.maxpool4(x)
         
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.averagepool(x)
        x = torch.flatten(x, start_dim=1)#展平
        x = self.droput(x)
        x = self.linear(x)
       # if init_Aux_logits:
        if self.training and init_Aux_logits:
            return x, Aux2, Aux1
        return x 
        
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        
        
class convRelu(nn.Module):
    def _init_(self, in_channels, out_channels):
        super(convRelu, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels),  
        self.relu = nn.ReLU(inplace=True),
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
    
class Inception(nn.Module):
    def _init_(self, in_channels, channels1_1, channels3_3red,channels3_3,channels5_5red,channels5_5,pool_proj):
        super(convRelu, self).__init__() 
        
        self.part1 = convRelu(in_channels,channels1_1,kernel_size=1,padding=0,stride=1 )
        
        self.part2 = nn.Sequential(
                convRelu(in_channels, channels3_3red, kernel_size=1, padding=0, stride=1 ),
                convRelu(channels3_3red, channels3_3, kernel_size=3, padding=1, stride=1 )
                )
        
        self.part3 = nn.Sequential(
                convRelu(in_channels, channels5_5red, kernel_size=1, padding=0, stride=1 ),
                convRelu(channels5_5red, channels5_5, kernel_size=5, padding=2, stride=1 )
                )
        
        self.part4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
                convRelu(in_channels, pool_proj, kernel_size=1, padding=0, stride=1 )
                )
        
    def forward(self, x):
        part1 = self.part1(x)
        part2 = self.part2(x)
        part3 = self.part3(x)
        part4 = self.part4(x)      
        outputs = [part1, part2, part3, part4]        
        return torch.cat(outputs, 1)#在channels上拼接
    
class Inceptionout(nn.Module):
    def _init_(self, num_classes, in_channels):
        super(Inceptionout, self).__init__() 
        self.averagepool = nn.AdaptiveAvgPool2d(kernel_size=5, stride=3), 
        self.conv4 = convRelu(in_channels, 128, kernel_size=1, stride=1, padding=0),#output 4*4*128=2048 展平
        
        self.linear1 = nn.Linear(2048, 1024),
        self.relu = nn.ReLU(inplace=True),
        self.droput = nn.Dropout(p=0.7),
        self.linear2 = nn.Linear(2048, 1024),
    
    def forward(self, x):
        x = self.averagepool(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1)#output=2048
        x = F.droput(x, 0.7, training = self.training)
        #x = self.droput(x)
        x = self.linear1(x)
        x = self.relu(x) 
        x = F.droput(x, 0.7, training = self.training)
        #x = self.droput(x)
        x = self.linear2(x)
        return x
    
        
    
              
        
        
            
        