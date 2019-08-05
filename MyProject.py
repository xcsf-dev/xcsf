import xcsf.xcsf as xcsf

print(xcsf.greet())

print(xcsf.square(4))

xcs = xcsf.XCS("sine_1var", 5000)

print("POP_SIZE = " + str(xcs.get_pop_num())) 
