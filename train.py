import torch
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from model import *
from data import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_printoptions(precision=20)

batch_size = 20 # need to be bigger than 7 to calculate SSIM
learning_rate = 0.001
epochs = 100
record = True

H=256
W=448
# B0
stage_num = 4
input_channels = 6
patch_size = (4, 2, 2, 2)
embed_dim = (32, 64, 160, 256)
block_num = (2, 2, 2, 2)
sr_ratio = (8, 4, 2, 1)
mlp_ratio = (8, 8, 4, 4)
num_head = (1, 2, 5, 8)
drop_rate = 0.3
atten_drop_rate = 0.3
drop_path_rate = 0.3

# # B1
# stage_num = 4
# input_channels = 6
# patch_size = (4, 2, 2, 2)
# embed_dim = (64, 128, 320, 512)
# block_num = (2, 2, 2, 2)
# sr_ratio = (8, 4, 2, 1)
# mlp_ratio = (8, 8, 4, 4)
# num_head = (1, 2, 5, 8)
# drop_rate = 0.3
# atten_drop_rate = 0.3
# drop_path_rate = 0.3

model = VFImodel(H=H, W=W,
                 stage_num=stage_num,
                 input_channels=input_channels,
                 patch_size=patch_size,
                 embed_dim=embed_dim,
                 block_num=block_num,
                 sr_ratio=sr_ratio,
                 mlp_ratio=mlp_ratio,
                 num_head=num_head,
                 drop_rate=drop_rate,
                 atten_drop_rate=atten_drop_rate,
                 drop_path_rate=drop_path_rate
                 ).to(device)

# state_dict = torch.load('model_pth_1/model_iter10000.pth')
# model.load_state_dict(state_dict)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.L1Loss().to(device)

train_dataset = dataset(train=True)
test_dataset = dataset(train=False, max_num=100)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if record:
    writer = SummaryWriter()

# torch.autograd.set_detect_anomaly(True)
model.train()
total_iteration_num = 0
for i in range(epochs):
    print('Epoch:{}'.format(i + 1))
    for x1, x2, x3 in train_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)

        x2_pre = model(x1, x3)
        train_loss = loss(x2_pre, x2)
        optim.zero_grad()
        train_loss.backward()
        optim.step()

        total_iteration_num += 1
        if total_iteration_num % 10 == 0 and x1.shape[0] >= 7: # >= 7 to calculate SSIM
            x2_pre = torch.clamp(x2_pre, min=0, max=1)
            mse = torch.mean((x2_pre - x2) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            var1 = Variable(x2, requires_grad=False).cpu().numpy()
            var2 = Variable(x2_pre, requires_grad=False).cpu().numpy()
            ssim_val = ssim(var1, var2, channel_axis=1)
            print("Epoch:{}, Total iterations:{}, Train loss:{}, PSNR:{}, SSIM:{}"
                  .format(i + 1, total_iteration_num, train_loss, psnr, ssim_val))
            if record:
                writer.add_scalar('train_loss', train_loss, total_iteration_num)
                writer.add_scalar('train_PSNR', psnr, total_iteration_num)
                writer.add_scalar('train_SSIM', ssim_val, total_iteration_num)

        if total_iteration_num % 100 == 0:
            model.eval()
            total_test_loss = 0
            total_psnr = 0
            total_ssim = 0
            test_batch_num = 0
            example_origin = 0
            example_pre = 0
            with torch.no_grad():
                for j, (test_x1, test_x2, test_x3) in enumerate(test_loader):
                    test_x1 = test_x1.to(device)
                    test_x2 = test_x2.to(device)
                    test_x3 = test_x3.to(device)
                    test_x2_pre = model(test_x1, test_x3)
                    test_loss = loss(test_x2_pre, test_x2)
                    total_test_loss += test_loss

                    if record and j == 0:
                        test_x2_pre = torch.clamp(test_x2_pre, min=0, max=1)
                        example_origin = test_x2.cpu().detach()
                        example_pre = test_x2_pre.cpu().detach()
                        example_origin /= example_origin.max()
                        example_pre /= example_pre.max()
                        save_image(example_pre, 'test_photo/img_iteration{}_pre.png'.format(total_iteration_num), normalize=True)
                        save_image(example_origin, 'test_photo/img_iteration{}_true.png'.format(total_iteration_num), normalize=True)


                    if test_x2.shape[0]>=7:
                        test_batch_num += 1

                        mse = torch.mean((test_x2_pre - test_x2) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        total_psnr += psnr

                        var1 = Variable(test_x2, requires_grad=False).cpu().numpy()
                        var2 = Variable(test_x2_pre, requires_grad=False).cpu().numpy()
                        ssim_val = ssim(var1, var2, channel_axis=1)
                        total_ssim += ssim_val

            print("Epoch:{}, Total iterations:{}, Total test loss:{}, Average PSNR:{}, SSIM:{}"
                  .format(i + 1, total_iteration_num, total_test_loss, total_psnr/test_batch_num, total_ssim/test_batch_num))
            if record:
                writer.add_scalar('total_test_loss', total_test_loss, total_iteration_num)
                writer.add_scalar('test_PSNR', total_psnr/test_batch_num, total_iteration_num)
                writer.add_scalar('test_SSIM', total_ssim/test_batch_num, total_iteration_num)
                if total_iteration_num % 1000 == 0:
                    torch.save(model.state_dict(), 'model_pth/model_iter{}.pth'.format(total_iteration_num))
            model.train()

    if i == 95:
        learning_rate = 1e-4
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print('Reset learning rate as {}.'.format(learning_rate))
