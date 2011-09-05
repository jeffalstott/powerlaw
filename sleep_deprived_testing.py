from scipy.stats import linregress
import sqlite3
from numpy import asarray, empty
import matplotlib.pyplot as plt

database = '/work/imagingA/jja34/Results'
threshold_level = .99
time_scale = 1
band_name = 'broad'
variable = 'size_events'
values = (variable, time_scale, threshold_level, band_name)

conn = sqlite3.connect(database)
join_string = 'Subjects NATURAL JOIN Experiments NATURAL JOIN Recordings_Raw JOIN Recordings_Filtered ON Recordings_Raw.recording_id=Recordings_Filtered.recording_id NATURAL JOIN Avalanche_Analyses NATURAL JOIN Fit_Statistics'

d_2 = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK1'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and visit_number=2" % join_string, values).fetchall())

d_3 = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK1'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and visit_number=3" % join_string, values).fetchall())

both_visits = list(set(d_2[:,0]) & set(d_3[:,0]))

v2 = empty(len(both_visits))
v3 = empty(len(both_visits))

for i in range(len(both_visits)):
    subject = both_visits[i]
    v2[i] = d_2[d_2[:,0]==subject, 1]
    v3[i] = d_3[d_3[:,0]==subject,1]

slope, intercept, R, p, stderr = linregress(v2, v3)
print 'Slope = '+str(slope)
print 'R = '+str(R)
plt.scatter(v2, v3)
plt.plot((0,1), [slope*x + intercept for x in [0, 1]])


d_r = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK2'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and rest='rested'" % join_string, values).fetchall())
d_sd = asarray(conn.execute("SELECT number_in_group,KS,parameter1_value,p FROM %s WHERE group_name='GSK2'\
        AND distribution='power_law' AND variable = ? and time_scale= ? and threshold_level= ? \
        and band_name= ? and rest='sleep_deprived'" % join_string, values).fetchall())

both_visits = list(set(d_sd[:,0]) & set(d_r[:,0]))

sd = empty(len(both_visits))
r = empty(len(both_visits))

for i in range(len(both_visits)):
    subject = both_visits[i]
    sd[i] = d_sd[d_sd[:,0]==subject, 1]
    r[i] = d_r[d_r[:,0]==subject,1]

slope, intercept, R, p, stderr = linregress(r, sd)
plt.figure()
print 'Slope = '+str(slope)
print 'R = '+str(R)
plt.scatter(r, sd)
plt.plot((0,1), [slope*x + intercept for x in [0, 1]])

conn.close()

