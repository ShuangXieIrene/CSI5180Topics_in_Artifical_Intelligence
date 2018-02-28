import matplotlib.pyplot as plt

plt.ylabel("Accuracy")
plt.ylim(0.9,1.0) # Define the interval of y-axis
plt.xlabel("Disambiguation Confidence")
x_value =[0.2,0.3,0.4,0.5,0.7] 

KNN_Acc =[0.994,0.992,0.980,0.982,0.965]
RF_Acc  =[0.994,0.994,0.986,0.982,0.943]
Naive_B =[0.959,0.960,0.973,0.936,0.933]
plt.plot(x_value, KNN_Acc, marker='o', color='r', label='KNN')#KNN
plt.plot(x_value, RF_Acc, marker='^', color='g', label='Random Forest') #RF
plt.plot(x_value, Naive_B, marker='*', color='b', label='Naive_Bayes')#NBB
plt.legend(loc='upper right')
plt.show()
