import math
import numpy as np
from sklearn import metrics
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def psnr_park(mse):
    return 10 * math.log10(1 / mse)


def anomaly_score(psnr, max_psnr, min_psnr):
    return (psnr - min_psnr) / (max_psnr - min_psnr)

def showimg(fig):
    img1 = mpimg.imread("C:\\College work\\astnet-main\\ASTNet\\datasets\\ped2\\testing\\frames\\01\\130.jpg")
    fig.add_subplot(221)
    plt.title('Anomaly')
    plt.imshow(img1)

    img2 = mpimg.imread("C:\\College work\\astnet-main\\ASTNet\\datasets\\ped2\\testing\\frames\\01\\015.jpg")
    fig.add_subplot(223)
    plt.title('No Anomaly')
    plt.imshow(img2)

def animate(i, psnr_list, fig):
    y = []
    x = []
    b=[]
    for k, psnr_sublist in enumerate(psnr_list): 
            a=[]
            for j, psnr_value in enumerate(psnr_sublist):  
                score = anomaly_score(psnr_value, np.max(psnr_sublist), np.min(psnr_sublist))
                x.append(j)
                y.append(score)
                if score < 0.5:
                    a.append(f'frame no {j} - anomoly')
                else:
                    a.append(f'frame no {j} - no anomoly')
            b.append(a)
            #fig.add_subplot(222)
            plt.title('graph')
            plt.cla()
            plt.plot(x, y)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            plt.tight_layout()

    file_path = './output.txt'

        # Open the file in write mode
    with open(file_path, 'w') as file:
            # Iterate through each sublist
        for sublist in b:
            for item in sublist:
                file.write(item + '\n')




def calculate_auc(config, psnr_list, mat):

    
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process

    scores = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.int32)

    fig = plt.figure()
    #showimg(fig)
    
    ani = FuncAnimation(plt.gcf(), animate, fargs=(psnr_list, fig,), interval=1000)

    plt.show()

    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list[i]), np.min(psnr_list[i]))
        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)
    assert scores.shape == labels.shape, f'Ground truth has {labels.shape[0]} frames, BUT got {scores.shape[0]} detected frames!'
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr