D_set={}

# Directory
D_set['name_phosphor']='material'
D_set["name"]="measurement.h5"

# everything in µs
nbImages=3 # number of accumulated pictures are taken
D_set['nbr_shots_total']= nbImages
D_set['delay_cam_start']= 150e-6
D_set['delay_led_end']= 5e-6
D_set['time_betw_exp'] = 200e-6

D_set['readout']= 15000*10**(-6)
D_set['timeout']= 4000 

tau = 100e-6
D_set['t_on_1']=tau
D_set['t1']=tau/2 
D_set['t2_dlr']=tau  
D_set['delta_t']=tau/2 

D_set['t2_start']=0e-6
D_set['t1_end']=0e-6
D_set['number_exposure']= 100
D_set['vval'] = 0
D_set['cycle_delay'] = 0.0 #s
D_set['mode'] = "RLD"
D_set["fit_type"] = "Oxygen"


#computed
D_set['delay_cam_end']= round(D_set['time_betw_exp'] - 2 * D_set["delta_t"] - D_set["delay_led_end"], 9)
D_set['parameters_to_send']=str(D_set['number_exposure'])+','+str(int(D_set['delta_t']*10**(6)))+','+str(int(D_set['t2_dlr']*10**(6)))+','+str(int(D_set['t_on_1']*10**(6)))+','+str(int(D_set['delay_led_end']))+','+str(int(D_set['delay_cam_end']))+','+str(int(D_set['vval']))+','+str(int(D_set['t2_start']*10**(6)))+' '
D_set['t_G1']=(D_set['t_on_1']+D_set['delay_led_end']+D_set['delta_t']+D_set['delta_t']+D_set['delay_cam_end'])
D_set['t_G2']=(D_set['t_on_1']+D_set['delay_led_end']+D_set['delta_t']+D_set['delta_t']+D_set['delay_cam_end'])
D_set['fps']=1/(D_set['number_exposure']*(D_set['t_G1']+D_set['t_G2'])+2*D_set['readout'])

D_set['image_acq_time_back1']= (D_set['number_exposure'] * (D_set['delay_cam_start']+D_set['delta_t'])+D_set['readout']) #, delay to wait to obtain the image gate1 
D_set['image_acq_time_back2']= (D_set['number_exposure'] * (D_set['delay_cam_start']+D_set['delta_t'])+D_set['readout']) #, delay to wait to obtain the image gate2 
D_set['image_acq_time1']= D_set['number_exposure'] * D_set['t_G1']+D_set['readout']  #s
D_set['image_acq_time2']= D_set['number_exposure'] * D_set['t_G2']+D_set['readout'] #s

D_set['x_min']=0
D_set['x_max']=2464
D_set['y_min']=0
D_set['y_max']=2056

D_set['ROI']=[D_set['x_min'],D_set['x_max'],D_set['y_min'],D_set['y_max']]

GUI_params = {
    "delta_t": D_set['delta_t'],
    "t2_dlr": D_set['t2_dlr'],
    "t_on_1": D_set['t_on_1'],
    "t1_start": D_set['delay_led_end'],
    "Exposures": D_set['number_exposure'],
    "Number of Images": D_set['nbr_shots_total'],
    "Time betw. Exposures": D_set['time_betw_exp'],
    "Voltage": 23.995 + 0.006146 * D_set['vval'], 
    "t2_start": D_set['t2_start'],
    "cycle_delay": D_set['cycle_delay']
}