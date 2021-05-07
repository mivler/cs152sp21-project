import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torchvision import transforms
from random import randint
from PIL import Image
import os

'''
**pAIcasso: An AI Artist**

Press below to generate art
'''


art_styles = ["Abstract Art", "Contemporary Art", "Cubist Art", "Gongbi Art", "Impressionist Art", "Kente Art", "Min-hwa Art", "Modern Art", "Mughal Art", "Surrealist Art", "Xieyi Art"]


class Generator(nn.Module):
    '''
    Generator model that includes an initial linear model to handle the 101th piece
    of data that represents the target style
    Generator then has 7 convolutional layers to create the image
    '''
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(101, 100)
        
        self.ConvSet1 = nn.Sequential(
                            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(True)
                        )
        
        self.ConvSet2 = nn.Sequential(
                            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True)
                        )

        self.ConvSet3 = nn.Sequential(
                            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True)
                        )
        
        self.ConvSet4 = nn.Sequential(
                            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True)
                        )
        
        self.ConvSet5 = nn.Sequential(
                            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True)
                        )
        
        self.ConvSet6 = nn.Sequential(
                            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True)
                        )


        self.ConvSet7 = nn.Sequential(
                            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                            nn.Tanh()
                        )

    def forward(self, X):
        x = self.linear1(X)
        Y = self.ConvSet1(x.unsqueeze(2).unsqueeze(3))
        y = self.ConvSet2(Y)
        Z = self.ConvSet3(y)
        z = self.ConvSet4(Z)
        W = self.ConvSet5(z)
        w = self.ConvSet6(W)
        V = self.ConvSet7(w)
        return V


def rand_generator_input():
    '''
    Function to generate random input of size [batch_size, 101, 1, 1] 
    The data consists of random noise and a target style piece appended to
    the 100 items of random noise
    '''
    # create batches separately then concatenate them all together 
    tensors = []
    for i in range(2):
        noise = torch.randn(101, 1, 1)
        target_category = randint(0, len(art_styles)-1)
        noise[100] = target_category
        tensors.append(noise)
    return torch.stack(tensors)


def save_generated_image(style_num):
    img_name = str(art_styles[style_num]) + "Generated.jpg"
    val = torch.min((target_labels == style_num).nonzero(as_tuple=True)[0])
    img = generated_images[val.item()]
    utils.save_image(img, img_name)
    
    # show image
    plt.imshow(generated_images[val.item()].cpu().permute(1,2,0))


generator = Generator()
generator.load_state_dict(torch.load("generator300.pkl", map_location=torch.device('cpu')))
generator.eval()

def make_image():
    noises = rand_generator_input().squeeze()
    generated_image = generator(noises)[0]
    return generated_image

generate_condition = st.button("Generate Art!")
if generate_condition:
    generated_image = make_image()
    image = transforms.ToPILImage()(generated_image).convert("RGB")
    st.image(image)