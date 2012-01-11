def alpha_sigma(user_options):
    import Helix_database as db

    x = db.Avalanche.sigma_events
    x_name = 'Sigma (Events)'
    x_range = (0,2)
    y = db.Fit.Kappa
    y_name = 'Kappa'
    y_range = (.8,1.2)
    color = db.Avalanche.time_scale
    color_name = 'Time Scale'
    import operator as op
    options = {'Avalanche.threshold_level': 10,
            'Avalanche.threshold_mode': 'Likelihood',
            'Avalanche.time_scale': (10.0, op.lt),
            'Filter.downsampled_rate': 200,
            'Filter.band_name': 'high-gamma',
            'Fit.fixed_xmax': True,
            'Fit.fixed_xmin': None,
            'Fit.distribution': 'power_law',
            'Fit.xmax': None,
            'Fit.xmin': 1,
            'Fit.variable': 'size_events',
            'Sensor.sensor_type': 'ECoG',
            'Subject.name': 'K1',
            'Task.eyes': None}
    options.update(user_options)

    data = db.compare(y, x, color, db.Task.type, db.Experiment.visit_number, **options)

    x_label = str(x).split('.')[-1]
    y_label = str(y).split('.')[-1]
    c_label = str(color).split('.')[-1]

    x_data = data[x_label]
    y_data = data[y_label]
    c_data = data[c_label]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
    sax = ax.scatter(x_data,y_data, c=c_data, cmap='jet')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if 'appa' in y_name:
        ax.plot(plt.xlim(), [1,1])
    elif 'lpha' in y_name:
        ax.plot(plt.xlim(), [1.5,1.5])
    if 'igma' in x_name:
        ax.plot([1,1], plt.ylim())
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#axins1 = inset_axes(ax, width="5%", # width = 10% of parent_bbox widt
#        height="90%", # height : 50%
#        loc=1)
    from numpy import unique
    cb =fig.colorbar(sax, ticks=unique(c_data), fraction=.08, pad=0)
    cb.set_label(color_name)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = ''
    for x in options.keys():
        if type(options[x])==tuple:
            textstr = textstr+x.split('.')[-1]+': '+str(options[x][0])+'\n'
        else:
            textstr = textstr+x.split('.')[-1]+': '+str(options[x])+'\n'
    ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=5, verticalalignment='top', bbox=props)



    ax1 = fig.add_subplot(223)
    ax1.set_xlim(0, max(c_data))
    ax1.set_ylim(0, max(x_data))
    ax1.set_xlabel(color_name)
    ax1.set_ylabel(x_name)

    ax2 = fig.add_subplot(224)
    ax2.set_xlim(0, max(c_data))
    ax2.set_ylim(min(y_data), max(y_data))
    ax2.set_xlabel(color_name)
    ax2.set_ylabel(y_name)

    conditions = unique(data[['type', 'visit_number']])
    from numpy import where
    for c in conditions:
        ind = where(data[['type', 'visit_number']]==c)
        order = c_data[ind].argsort()
        ax1.plot(c_data[ind][order], x_data[ind][order])
        ax2.plot(c_data[ind][order], y_data[ind][order])



    file_name = textstr.replace(': ', '-')
    file_name = file_name.replace('\n', ' ')
    fig.savefig('/home/alstottj/Figures/'+file_name+'.png')
