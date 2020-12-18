import os, time, pickle, argparse, networks, utils
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='project_name',  help='')
parser.add_argument('--src_data', required=False, default='src_data_path',  help='sec data path')
parser.add_argument('--tgt_data', required=False, default='tgt_data_path',  help='tgt data path')
parser.add_argument('--vgg_model', required=False, default='pre_trained_VGG19_model_path/vgg19.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--pre_trained_model', required=True, default='pre_trained_model', help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=False, default='image_dir', help='test image path')
parser.add_argument('--output_image_dir', required=False, default='output_image_dir', help='output test image path')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if torch.cuda.is_available():
    G.load_state_dict(torch.load(args.pre_trained_model),False)
else:
    # cpu mode
    G.load_state_dict(torch.load(args.pre_trained_model, map_location=lambda storage, loc: storage))
G.to(device)

src_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
# image_src = utils.data_load(os.path.join(args.image_dir), 'test', src_transform, 1, shuffle=True, drop_last=True)

# with torch.no_grad():
#     G.eval()
#     for n, (x, _) in enumerate(image_src):
#         x = x.to(device)
#         G_recon = G(x)
#         result = torch.cat((x[0], G_recon[0]), 2)
#         path = os.path.join(args.output_image_dir, str(n + 1) + '.png')
#         plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

source_dir = 'photo'
ani_dir = ['five_centimeters', 'papurika', 'spirited_away', 'the_girl', 'violet']
if(not os.path.exists(os.path.join(args.name + '_results'))):
            os.mkdir(os.path.join(args.name + '_results'))
with torch.no_grad():
    for ani_name in ani_dir:
        if(not os.path.exists(os.path.join(args.name + '_results', ani_name))):
            os.mkdir(os.path.join(args.name + '_results', ani_name))
        test_dir = ani_name
        test_loader_trans = utils.data_load(os.path.join('data', test_dir), 'test', src_transform, 8, shuffle=False, drop_last=True)
        test_loader_src = utils.data_load(os.path.join('data', source_dir), 'test', src_transform, 1, shuffle=False, drop_last=True)
        G.eval()
        y = test_loader_trans
        for (y,_) in (test_loader_trans):
            break
        # print(type(test_loader_trans),test_loader_trans)
        # for n, ((x, _), (y, _)) in enumerate(zip(test_loader_src, test_loader_trans)):
        for n, (x,_) in enumerate((test_loader_src)):
            x = x.to(device)
            y = y.to(device)
            G_recon = G(x, y)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', ani_name, args.name + '_test_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            print("EPOCH "+str(n)+" FINISH!")
            # if n == 4:
            #     break
        path_ = os.path.join(args.name + '_results', ani_name, args.name + '_test_' + 'tgt.png')
        plt.imsave(path_, (torchvision.utils.make_grid(y,4).cpu().numpy().transpose(1, 2, 0) + 1) / 2)

print("All Works Done!")