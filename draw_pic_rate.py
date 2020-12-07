from matplotlib import pyplot as plt
# 画数据分布图
# 设置中文字体
font={'family':'Microsoft Yahei','weight':'bold'}
plt.rc('font',**font)
plt.bar([1,2],[1341, 3876], color = 'black')
plt.xticks([1,2],['NORMAL','PNEUMONIA'])
plt.show()