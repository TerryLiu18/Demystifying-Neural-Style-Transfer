import torch
import torch.optim as optim
from torchvision import transforms, models

import skimage.transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import glob

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = im_transform(img).unsqueeze(0)
    return img

def convert_img(tensor):
    img = tensor.to("cpu").clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1,2,0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = np.clip(img, 0, 1)
    return img

def get_features(model, img):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1'}
    features = {}
    x = img
    for name,layer in enumerate(model.features):
        x = layer(x)
        name = str(name)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    batch_size, channel, width, height = tensor.size()
    features = tensor.view(batch_size * channel, width * height)
    G = torch.mm(features,features.t())
    return G.reshape(batch_size, channel, channel)


def main(style_name, 
        content_name,
        learning_rate,
        iteration,
        alpha,beta,
        down_model):
    if down_model:
        torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', model_dir='./models/')
    vgg = models.vgg19()
    vgg.load_state_dict(torch.load("./models/vgg19-dcbb9e9d.pth"))
    for param in vgg.parameters():
        param.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device).eval()

    style_image_list = []
    for file in glob.glob('./data/style/{}*'.format(args.style)):
        style_image_list.append(file)

    content_image_list = []
    for file in glob.glob('./data/content/{}*'.format(args.content)):
        content_image_list.append(file)

    style_weight = {
        'conv1_1': 1,
        'conv2_1': 1,
        'conv3_1': 1,
        'conv4_1': 1,
        'conv5_1': 1
    }
    
        # stylization
    for i_style in range(len(style_image_list)):
        style_img_name = style_image_list[i_style].split('/')
        style_img_name = style_img_name[-1].split('.')
        style_img_name = style_img_name[0].split('\\')[-1]
        for i_content in range(len(content_image_list)):
            
            content_img_name = content_image_list[i_content].split('/')
            content_img_name = content_img_name[-1].split('.')
            content_img_name = content_img_name[0].split('\\')[-1]
            
            style = load_img(style_image_list[i_style]).to(device)
            content = load_img(content_image_list[i_content]).to(device)
            
            # resize
            content_shape = content.shape
            style = transforms.Resize((content_shape[2], content_shape[3]))(style)
            
            style_feature = get_features(vgg,style)
            content_feature = get_features(vgg,content)
            style_grams = {layer_name: gram_matrix(style_feature[layer_name]) for layer_name in style_feature}

            input_img = content.clone().requires_grad_(True).to(device)
            optimizer = optim.LBFGS([input_img], lr=learning_rate)
            
            # begin stylization
            print('processing content: {}, style: {}'.format(content_img_name, style_img_name))
            
            I = [1]
            while I[0]<iteration+1:
                def closure():
                    I[0]+=1
                    optimizer.zero_grad()
                    input_feature = get_features(vgg,input_img)
                    content_loss = torch.mean((input_feature["conv5_1"]-content_feature["conv5_1"])**2)
                    style_loss = 0
                    for layer in style_weight:
                        input_layer = input_feature[layer]
                        input_gram = gram_matrix(input_layer)
                        b,d,h,w = input_layer.size()
                        style_gram = style_grams[layer]
                        
                        layer_style_loss = style_weight[layer]*torch.mean((input_gram-style_gram)**2)
                        style_loss += layer_style_loss/(d*h*w)
                    total_loss = alpha*content_loss + beta*style_loss
                    total_loss.backward()
                    if I[0]%50 == 0:
                        print("Iteration:{:d}, content_loss:{:.4f}, style_loss:{:.4f}".format(I[0],content_loss.item(),style_loss.item()))
                    return total_loss
                optimizer.step(closure)
            output_img = convert_img(input_img)
            plt.imsave("./results/{}_{}.png".format(content_img_name, style_img_name), output_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='', help='style image name')
    parser.add_argument('--content', type=str, default='', help='content image name')
    parser.add_argument('--learning_rate','--lr', type=float, default=1)
    parser.add_argument('--iteration','--iter', type=int, default=400)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1e17)
    parser.add_argument('--down_model','--down', type=str_to_bool, default=True)
    args = parser.parse_args()

    main(style_name=args.style,
        content_name=args.content,
        learning_rate=args.learning_rate,
        iteration=args.iteration,
        alpha=args.alpha,beta=args.beta,
        down_model=args.down_model)
    

