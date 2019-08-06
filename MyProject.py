import xcsf.xcsf as xcsf

xcs = xcsf.XCS("../data/sine_1var", 5000)

print("pop_num = " + str(xcs.get_pop_num())) 

print("num_x_vars = " + str(xcs.get_num_x_vars())) 
print("num_y_vars = " + str(xcs.get_num_y_vars())) 

print("cond_type = "+str(xcs.get_cond_type()))
print("pred_type = "+str(xcs.get_pred_type()))

xcs.fit()
