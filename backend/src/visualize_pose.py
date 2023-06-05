import cv2
import tqdm
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg

import file_utils as fu


if __name__ == '__main__':
    fu.check_cwd()
    np.random.seed(0)
    paths = glob('../data/pose/ACCAD/*/*.npy')
    path = np.random.choice(paths)
    pose_data = np.load(path)
    pose_data -= pose_data[:,0:1,:]
    rot_matrix = np.array([[-1,0,0],[0,0,-1],[0,1,0]])
    pose_data = np.matmul(rot_matrix, pose_data[:,:,:,None])
    pose_data = pose_data[:,:,:,0]
    # output pose video
    fig = plt.figure()
    # canvas = FigureCanvasAgg(fig)
    H, W = 960, 1280
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    z_min, z_max = -1, 1
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter('../data/pose.mp4', fourcc, 20, (W, H))
    idx = 0
    for poi in range(52):
        ax = plt.axes(projection='3d')
        # plot body
        ax.scatter(pose_data[idx,:22,0], pose_data[idx,:22,1],
            pose_data[idx,:22,2], color='blue', s=1)
        # plot left hand 
        ax.scatter(pose_data[idx,22:37,0], pose_data[idx,22:37,1],
            pose_data[idx,22:37,2], color='orange', s=1)
        # plot right hand 
        ax.scatter(pose_data[idx,37:,0], pose_data[idx,37:,1],
            pose_data[idx,37:,2], color='green', s=1)
        # plot point of interest
        ax.scatter(pose_data[idx,poi:poi+1,0], pose_data[idx,poi:poi+1,1],
            pose_data[idx,poi:poi+1,2], color='red', s=10)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.zaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_box_aspect((x_max-x_min,y_max-y_min,z_max-z_min))
        ax.set_title(f'{poi}')
        plt.show()
        # exit()
    #     buffer, (w, h) = canvas.print_to_buffer()
    #     img = np.frombuffer(buffer, np.uint8).reshape((h, w, 4))
    #     img = cv2.resize(img, (W, H))
    #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    #     video_writer.write(img)
    #     fig.clear()
    # video_writer.release()