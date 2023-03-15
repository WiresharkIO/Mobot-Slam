import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
	This code describes the kinematic of a Bicycle-model and based on that evolves a "Figure Eight" trajectory. This trajectory can be replaced with any, by changing the velocity (v_data) and the steering 		angle(delta), steering angle rate (w) in the __main__
"""
class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22
        
        self.sample_time = 0.01
        
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
    def step(self, v, w):
        if w > 0:
            w = min(w, self.w_max)
        else:
            w = max(w, -self.w_max)
 
        xc_dot = v * np.cos(self.theta + self.beta)
        yc_dot = v * np.sin(self.theta + self.beta)
        theta_dot = (v / self.L) * (np.cos(self.beta) * np.tan(self.delta))
        delta_dot = w
        self.beta = np.arctan(self.lr * np.tan(self.delta) / self.L)
        
        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.theta += theta_dot * self.sample_time 
        self.delta += delta_dot * self.sample_time
        
if __name__ == "__main__":
	model = Bicycle()  
	      
	sample_time = 0.01
	time_end = 30
	
	model.reset()
	
	t_data = np.arange(0,time_end,sample_time)
	x_data = np.zeros_like(t_data)
	y_data = np.zeros_like(t_data)
	v_data = np.zeros_like(t_data)
	w_data = np.zeros_like(t_data)


	delta = 0.95*np.arctan(2/8)
	v_data[:] = (2*np.pi*2* 8) / (time_end)

	for i in range(t_data.shape[0]):
	    x_data[i] = model.xc
	    y_data[i] = model.yc
	    
	    if i <= t_data.shape[0]/8:
	    	if model.delta < delta:
	    		model.step(v_data[i], model.w_max)
	    		w_data[i] = model.w_max
	    	else:
	    		model.step(v_data[i], 0)
	    		w_data[i] = 0
				
	    elif i <= (5.1*t_data.shape[0])/8:
	    	if model.delta > -delta:
	    		model.step(v_data[i], -model.w_max)
	    		w_data[i] = -model.w_max
	    	else:
	    		model.step(v_data[i], 0)
	    		w_data[i] = 0
		    
	    else:
	    	if model.delta < delta:
	    		model.step(v_data[i], model.w_max)
	    		w_data[i] = model.w_max
	    	else:
	    		model.step(v_data[i], 0)
	    		w_data[i] = 0 

	plt.axis('equal')
	plt.plot(x_data, y_data)
	plt.show()
