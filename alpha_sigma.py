import Helix_database as db

options = {}
options['Avalanche.time_scale']=1
#options['Fit.xmin']=1
#options['Fit.xmax']=None

data = db.compare(db.Fit.Kappa, db.Avalanche.sigma_events, db.Avalanche.time_scale, **options)

import matplotlib.pyplot as plt
x_label = 'Time Scale'
x = data[:,1]
y_label = 'Kappa'
y = data[:,0]
plt.scatter(x,y)
plt.ylim(.9,1.1)
plt.plot(plt.xlim(), [1,1])
plt.xlim(0,3)
plt.plot([1,1], plt.ylim())
plt.draw()
plt.show()
