from scipy.stats import linregress
import sqlite3
from numpy import asarray, empty
import pylab as plt

database = '/work/imagingA/jja34/Results'
threshold_level = .99
time_scale = 1
band_name = 'beta'
variable = 'size_amplitudes'
values = (variable, time_scale, threshold_level, band_name)
output_measure = ('KS',1)

conn = sqlite3.connect(database)
join_string = 'Subjects NATURAL JOIN Experiments NATURAL JOIN Recordings_Raw JOIN Recordings_Filtered ON Recordings_Raw.recording_id=Recordings_Filtered.recording_id NATURAL JOIN Avalanche_Analyses NATURAL JOIN Fit_Statistics'

d_2 = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK1'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and visit_number=2" % join_string, values).fetchall())

d_3 = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK1'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and visit_number=3" % join_string, values).fetchall())

both_visits = list(set(d_2[:,0]) & set(d_3[:,0]))

v2 = []
v3 = []

for i in range(len(both_visits)):
    subject = both_visits[i]
    v2.append(d_2[d_2[:,0]==subject, output_measure[1]][0])
    v3.append(d_3[d_3[:,0]==subject, output_measure[1]][0])

slope, intercept, R, p, stderr = linregress(v2, v3)
print 'Slope = '+str(slope)
print 'R = '+str(R)
print 'p = '+str(p)
figure();
plt.scatter(v2, v3)
plt.plot((0,max(v2)), [slope*x + intercept for x in [0, max(v2)]])
plt.title('Control group, '+band_name+' band, '+str(threshold_level)+' percentile, '+str(time_scale)+' samples per bin')
plt.xlabel(output_measure[0]+', Visit 1')
plt.ylabel(output_measure[0]+', Visit 2')
plt.show()


d_r = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK2'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and rest='rested'" % join_string, values).fetchall())
d_sd = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK2'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and rest='sleep_deprived'" % join_string, values).fetchall())

both_visits = list(set(d_sd[:,0]) & set(d_r[:,0]))

sd = []
r = []

for i in range(len(both_visits)):
    subject = both_visits[i]
    sd.append(d_sd[d_sd[:,0]==subject, output_measure[1]][0])
    r.append(d_r[d_r[:,0]==subject,output_measure[1]][0])

slope, intercept, R, p, stderr = linregress(r, sd)
plt.figure()
print 'Slope = '+str(slope)
print 'R = '+str(R)
print 'p = '+str(p)
plt.scatter(r, sd)
plt.plot((0,max(r)), [slope*x + intercept for x in [0, max(r)]])
plt.title('Test group, '+band_name+' band, '+str(threshold_level)+' percentile, '+str(time_scale)+' samples per bin')
plt.xlabel(output_measure[0]+', Rested')
plt.xlabel(output_measure[0]+', Sleep deprived')
plt.show()

conn.close()

