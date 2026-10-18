import sys
sys.path.append('C:/XIMEA/API/Python/v3/')
from ximea import xiapi
import time
import h5py
import numpy as np

def take_one_image_ratio(D_set,  cam, stm, img, mode):
    if mode == "RLD" or mode == "RLD (overlapping)":
        x = 1
        y = 2
    else:
        x = 5
        y = 6
    try:
        # Gate 1
        t0 = time.time()
        stm.write(bytes(str(x), 'utf-8'))
        print("sending")
        time.sleep(D_set['image_acq_time1'])
        t1 = time.time()
        cam.get_image(img, timeout =D_set['timeout'])
        print("time")
        image_G1 = img.get_image_data_numpy()
        t_G1=float(img.tsSec + img.tsUSec*1e-6)
        # Gate 2
        stm.write(bytes(str(y), 'utf-8'))
        time.sleep(D_set['image_acq_time2'])
        cam.get_image(img, timeout =D_set['timeout'])
        image_G2 = img.get_image_data_numpy()
        t_G2=float(img.tsSec + img.tsUSec*1e-6)
    except :
        print("error")
        image_G1 = np.zeros((10000, 10000))
        image_G2 = np.zeros((10000, 10000))
        t_G1=np.nan
        t_G2=np.nan
    return image_G1, image_G2, t_G1, t_G2

def background(D_set, cam, stm, img):
    """
    Acquire background images for both gates.
    
    Parameters:
    -----------
    D_set : dict
        parameters dictionary
    cam : xiapi.Camera
        Camera object
    stm : serial.Serial
        Stm serial connection
    img : xiapi.Image
        Image object for storing acquisition
        
    Returns:
    --------
    image_background_1, image_background_2 : numpy.ndarray
        Background images for gate 1 and gate 2
    """
    L_image_background_1 = []
    L_image_background_2 = []
    
    stm.reset_input_buffer()
    
    # Acquire background images for gate 1
    for i in range(10):
        if i == 0:
            stm.write(bytes(str(3) + D_set['parameters_to_send'], 'utf-8'))
        else:
            stm.write(bytes(str(3), 'utf-8'))
        time.sleep(D_set['image_acq_time_back1'])
        cam.get_image(img, timeout=D_set['timeout'])
        L_image_background_1.append(img.get_image_data_numpy())
    
    L_image_background_1 = np.array(L_image_background_1)
    image_background_1 = np.mean(L_image_background_1, axis=0)
    
    # Acquire background images for gate 2
    for _ in range(10):
        stm.write(bytes(str(4), 'utf-8'))
        time.sleep(D_set['image_acq_time_back2'])
        cam.get_image(img, timeout=D_set['timeout'])
        L_image_background_2.append(img.get_image_data_numpy())
    
    # Convert to numpy array and average
    L_image_background_2 = np.array(L_image_background_2)
    image_background_2 = np.mean(L_image_background_2, axis=0)
    
    return image_background_1, image_background_2

def hdf5_read(filename, group_name):
    """
    Read datasets from a group in an HDF5 file.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    group_name : str
        Name of the group to read
        
    Returns:
    --------
    D_datasets : dict
        Dictionary of dataset_name: dataset_array pairs
    """
    with h5py.File(filename, 'r') as h5file:
        D_datasets = {}
        for dataset_name in h5file[group_name].keys():
            D_datasets[dataset_name] = h5file[group_name][dataset_name][:]
    return D_datasets

def hdf5_append(filename, group_name, D_datasets, chunks=True, compression='gzip', compression_opts=4):
    """
    Append data to an HDF5 file in a specified group.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    group_name : str
        Name of the group to create in the file
    D_datasets : list of tuples
        List of (dataset_name, dataset_array) tuples to save
    chunks : bool, optional
        Whether to use chunking
    compression : str, optional
        Compression algorithm to use
    compression_opts : int, optional
        Compression level
    """
    with h5py.File(filename, 'a') as h5file:
        if group_name in h5file:
            del h5file[group_name]
        data_group = h5file.create_group(group_name)
        data_group.attrs['description'] = 'matrices'
        for dataset_name, dataset_array in D_datasets:
            print(dataset_name)
            data_group.create_dataset(dataset_name, data=dataset_array, 
                                     chunks=chunks, compression=compression, 
                                     compression_opts=compression_opts)
        print(f'Data saved in file: {filename} in group: {group_name}')

# HARDWARE CONFIGURATION FUNCTIONS

def set_xiapi_parameters(number_exposure):
    """
    Initialize and configure the XIMEA camera.
    
    Parameters:
    -----------
    number_exposure : int
        Number of exposures to acquire per frame
        
    Returns:
    --------
    cam : xiapi.Camera
        Configured camera object
    """
    cam = xiapi.Camera()

    try:
        cam.close_device()
    except:
        pass

    time.sleep(0.1)

    cam.open_device()
    cam.set_imgdataformat('XI_MONO16')   # set image format
    cam.set_gpi_selector('XI_GPI_PORT2')   
    cam.set_gpi_mode('XI_GPI_TRIGGER')   # set trigger selector: Trigger controls the start of the exposure of one Frame.
    cam.set_trigger_selector('XI_TRG_SEL_EXPOSURE_ACTIVE')
    cam.set_exposure_burst_count(number_exposure)   # set the number of times of exposure in one frame
    cam.set_gpo_selector('XI_GPO_PORT3')   # set digital output
    cam.set_gpo_mode('XI_GPO_EXPOSURE_ACTIVE')
    cam.set_trigger_source('XI_TRG_EDGE_RISING')
    #cam.set_binning_vertical(2)
    #cam.set_width(256)
    #cam.set_height(256)
    
    # Set the rotation of the picture if needed
    horizontal_flip = cam.is_horizontal_flip()
    if horizontal_flip:
        cam.enable_horizontal_flip()
    vertical_flip = cam.is_vertical_flip() 
    if vertical_flip:
        cam.enable_vertical_flip()
    
    return cam

def compute_mean_ratio(filename, G1_ref=None, G2_ref=None):
    """Compute mean ratio from averaged, reference-corrected gates."""

    print(f" {filename}")

    with h5py.File(filename, "r") as f:
        gate_groups = sorted(g for g in f.keys() if g.startswith("gates_"))
        if not gate_groups:
            raise RuntimeError("No gate frames found")

    G1_accum = None
    G2_accum = None
    count = 0

    for g in gate_groups:
        gate_data = hdf5_read(filename, g)

        G1 = gate_data["G1"].astype(np.float64) 
        G2 = gate_data["G2"].astype(np.float64) 

        if G1_ref is not None:
            G1 -= G1_ref
        if G2_ref is not None:
            G2 -= G2_ref

        if G1_accum is None:
            G1_accum = np.zeros_like(G1)
            G2_accum = np.zeros_like(G2)

        G1_accum += G1
        G2_accum += G2
        count += 1

    G1_mean = G1_accum / count
    G2_mean = G2_accum / count

    Ratio_mean = G2_mean / (G1_mean + 1e-9)

    return Ratio_mean, G1_mean, G2_mean


def set_axis_labels(ax, fit_type, unit="%"):
    if fit_type == "Oxygen":

        if unit == "%":
            ax.set_xlabel("pO₂ (% air)")
        elif unit == "kPa":
            ax.set_xlabel("pO₂ (kPa)")
        elif unit == "mmol/L":
            ax.set_xlabel("[O₂] (mmol/L)")
        else:
            ax.set_xlabel("pO₂")

        ax.set_ylabel("Lifetime τ (µs)")

    elif fit_type == "Temperature":
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Lifetime τ (µs)")

    elif fit_type == "pH":
        ax.set_xlabel("pH")
        ax.set_ylabel("Lifetime τ (µs)")

        