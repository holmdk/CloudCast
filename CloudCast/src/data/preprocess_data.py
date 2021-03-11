"""
Preproccesing of images
"""
# Author: Andreas Holm <ahn@eng.au.dk>
import numpy as np

def pre_process_data(vals, nan_to_zero, normalize, restrict_classes_to_4, fill_value = np.nan):
    if np.isnan(fill_value):
        vals = vals.astype('float')
    vals[vals==0] = fill_value # nothing
    vals[vals==1] = fill_value # cloud-free land
    vals[vals==2] = fill_value # Cloud-free sea
    vals[vals==3] = fill_value # snow over land
    vals[vals==4] = fill_value # sea ice
    # vals[vals>250] = fill_value

    vals[vals>230] = fill_value

    if nan_to_zero:
        if np.isnan(fill_value):
            vals[np.isnan(vals)] = 0
        else:
            vals[vals < 0] = 0

        vals[vals > 0] = vals[vals > 0] - 4

    if restrict_classes_to_4:
        vals[vals==6] = 1 # fractional clouds are set to low-level cloud  --> CTTH is not available for clouds classified as fractional.
                          # however, fractional clouds = low level clouds, will in most cases match reality
                          #  as fractional clouds are typically small cumulus and stratocumulus clouds
                          # that is, clouds between 300m and 3km height

        # we define low-, mid- and high-level clouds
        vals[(vals > 0) & (vals < 3)] = 1 # low-level clouds
        vals[vals == 3] = 2               # mid-level clouds
        vals[(vals > 3)] = 3              # high-level clouds

    if normalize == True:
        if restrict_classes_to_4:
            vals = 2 * (vals / 3) - 1  # normalize to be between [-1, 1}
        else:
            vals = 2 * (vals / 10) - 1  # normalize to be between [-1, 1}
    return vals
