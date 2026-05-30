from libs import DatasetMagnetometerDetection

import torch

import matplotlib.pyplot as plt

import numpy

if __name__ == "__main__":

    dataset_root_path = "/users/michal/datasets/car_detection_2/"
    dataset = DatasetMagnetometerDetection(dataset_root_path + "Meranie_20_06_01-Lietavska_Lucka/01/")


    print("num samples ", len(dataset))

    
    model = torch.load("results/rnn_model_detector/model_9.pt",  weights_only=False, map_location='cpu')
    model.eval()


    for n in range(len(dataset)):

        #idx = numpy.random.randint(0, len(dataset)-1024)

        # take window, shif one by one
        x, y_gt = dataset[n*100]

        

        x_t = torch.from_numpy(x).float().unsqueeze(0)

        y_pred = model(x_t)
        y_pred = torch.nn.functional.sigmoid(y_pred)
        y_pred = y_pred.detach().numpy().squeeze(0)

        print(x.shape, y_gt.shape, y_pred.shape)
        # x.shape = (512, 3), y_gt.shape = (512, 1) y_pred.shape=(512, 1)


        plt.cla()
        plt.clf()

        plt.plot(x[:, 0], label="x", alpha=0.8)
        plt.plot(x[:, 1], label="y", alpha=0.8)
        plt.plot(x[:, 2], label="z", alpha=0.8)

        plt.plot(y_gt[:, 0], label="ground truth", color="red", lw=2.0)
        plt.plot(y_pred[:, 0], label="prediciton", color="blue", lw=2.0)

        plt.legend()
        plt.show()

        
        
