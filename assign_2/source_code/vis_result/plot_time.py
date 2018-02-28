import matplotlib.pyplot as plt

f,(ax,ax2) = plt.subplots(2,1,sharex=True)

plt.xlabel("Disambiguation Confidence")
ax.set_ylabel('Time_Consumption(s)')
ax2.set_ylabel('Time_Consumption(s)')

x_value = [0.2,0.3,0.4,0.5,0.7]
KNN_Train_Time = [0.065,0.109,0.008,0.022,0.018]
RF_Train_Time = [53.963,54.430,10.568,25.619,20.017]
NB_Train_Time = [0.223,0.225,0.029,0.076,0.055]
KNN_Test_Time = [23.769,23.295,5.126,6.566,6.740]
RF_Test_Time  = [0.397,0.445,0.163,0.193,0.178]
NB_Test_Time  = [0.025,0.025,0.002,0.006,0.004]

#Plot data
ax2.plot(x_value, KNN_Train_Time, marker='o', color='r', label='KNN_Train', linestyle='--')
ax2.plot(x_value, NB_Train_Time, marker='*', color='b', label='Naive_Bayes_Train', linestyle='--')
ax2.plot(x_value, RF_Test_Time, marker='^', color='g', label='Random Forest_Test')
ax2.plot(x_value, NB_Test_Time, marker='*', color='b', label='Naive_Bayes_Test')
ax.plot(x_value, RF_Train_Time, marker='^', color='g', label='Random Forest_Train', linestyle='--')
ax.plot(x_value, KNN_Test_Time, marker='o', color='r', label='KNN_Test')

#limit the view to different portions of the data
ax2.set_ylim(0, 0.5)
ax.set_ylim(5, 60)

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax2.legend(loc='upper right')
ax.legend(loc='upper right')

plt.show()
