from pyresample.geometry import AreaDefinition
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# Change colors for matplotlib CM
blues = mpl.cm.get_cmap('Blues', 12)
newcolors = blues(np.linspace(0, 1, 256))
newcolors[:, :] *= 0.9 # make colors less white
newcmp = mpl.colors.ListedColormap(newcolors)

def save_images_cartopy(prediction_video, ground_truth_video, target_times, batch, lines=False, high_res_map=True, path="/data/output/"):
    height = 128   # 18000 resolution
    width = 128

    # projection parameters
    lower_left_xy = [-855100.436345, -4942000.0]
    upper_right_xy = [1448899.563655, -2638000.0]
    area_def = AreaDefinition('areaD', 'Europe', 'areaD',
                              {'lat_0': '90.00', 'lat_ts': '50.00',
                               'lon_0': '5', 'proj': 'stere', 'ellps': 'WGS84'},
                              height, width,
                              (lower_left_xy[0], lower_left_xy[1],
                               upper_right_xy[0], upper_right_xy[1]))

    crs = area_def.to_cartopy_crs()

    create_video(prediction_video, 'Prediction', target_times, batch, crs, lines, high_res_map, path)
    create_video(ground_truth_video, 'Output', target_times, batch, crs, lines, high_res_map, path)


def create_video(video, name, target_times, batch, crs, lines=False, high_res_map=True, path="/data/output/"):
    # Create folder for batch images if not exist
    if not os.path.exists(path + 'batch_{}'.format(batch)):
        os.makedirs(path + 'batch_{}'.format(batch))

    for frame in np.arange(0, video.shape[1]):
        img = video[0, frame, :, :].detach().cpu().numpy()
        # renormalize for tanh
        #img = np.round((img + 1) * 14 / 2,0)
        img = img.astype('float')
        img[img==0] = np.nan  # no cloud we want to be transparent

        plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=crs)

        if lines:
            ax.coastlines()
            ax.gridlines()
            ax.set_global()
        if high_res_map:
            ax.background_img(name='BM', resolution='low')  # BM is a custom image collected from https://neo.sci.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG-TB

        plt.imshow(img, cmap=newcmp, transform=crs, extent=crs.bounds, origin='upper')
        plt.savefig(path + 'batch_{}/'.format(batch) + name + '_' + str(target_times[0, frame])[0:16] + '_frame_' + str(frame) + '.png')

        plt.close()


