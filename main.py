import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from UnetTest.module import Unet
from UnetTest.dataset import LiverDataset
from UnetTest.mIou import *
# 是否使用cuda
import PIL.Image as Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(i):
    from UnetTest import dataset
    imgs = dataset.make_dataset("./val")
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i], imgy[i]


def train_model(model, criterion, optimizer, dataload, num_epochs=21):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    # torch.save(model.state_dict(), './weights_%d.pth' % epoch)
    torch.save(model.state_dict(), 'modelUnet.pth')
    return model


# 训练模型
def train():
    model = Unet(3, 1).to(device)
    # batch_size = args.batch_size
    batch_size = 20
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("./train", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = train_model(model, criterion, optimizer, dataloaders,1)
    return model

# 显示模型的输出结果
def test():
    model = Unet(3, 1).to(device)  # unet输入是三通道，输出是一通道，因为不算上背景只有肝脏一个类别
    model.load_state_dict(torch.load("modelUnet.pth", map_location='cpu'))  # 载入训练好的模型
    liver_dataset = LiverDataset("./val", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()  # 开启动态模式

    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        num = len(dataloaders)  # 验证集图片的总数
        for x, _ in dataloaders:
            x = x.to(device)
            y = model(x)

            img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            mask = get_data(i)[1]  # 得到当前mask的路径
            miou_total += get_iou(mask, img_y)  # 获取当前预测图的miou，并加到总miou中
            plt.subplot(121)
            plt.imshow(Image.open(get_data(i)[0]))
            plt.subplot(122)
            plt.imshow(img_y)
            plt.pause(0.01)
            if i < num: i += 1  # 处理验证集下一张图
        print("图呢？？")
        plt.show()
        print('Miou=%f' % (miou_total / 20))


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    # 参数解析器,用来解析从终端读取的命令
    # parse = argparse.ArgumentParser()
    # # parse = argparse.ArgumentParser()
    # parse.add_argument("--action", type=str, help="train or test", default="train")
    # parse.add_argument("--batch_size", type=int, default=1)
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    # args = parse.parse_args()

    # train
    model = train()        #测试时，就把此train()语句注释掉

    # test()
    # args.ckp = r"modelUnet.pth"
    test()
