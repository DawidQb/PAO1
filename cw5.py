from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt


def get_mean_and_var(regex):
    file_list = glob.glob('reduced/' + regex + '*.fits')

    means = []
    datas = []

    for file_name in file_list:
        hdu_list = fits.open(file_name, memmap=False)
        image_data = hdu_list[0].data
        datas.append(image_data)
        means.append(np.mean(image_data))
        hdu_list.close()
    datas = np.array(datas)
    full_mean = np.mean(datas)

    mean_flat = np.zeros(datas[9].shape)
    var_flat = np.zeros(datas[9].shape)
    for i in range(len(datas)):
        datas[i] = datas[i]/means[i]*full_mean
        mean_flat += datas[i]
    N = len(datas)
    mean_flat = mean_flat/N
    for i in range(len(datas)):
        var_flat += (datas[i]-mean_flat)**2
    var_flat = var_flat/(N-1)
    y1 = 840
    y2 = 940
    x1 = 1070
    x2 = 1170
    c_mean = np.mean(mean_flat[y1:y2, x1:x2])
    var_mean = np.mean(var_flat[y1:y2, x1:x2])
    print(regex, c_mean, var_mean)
    return c_mean, var_mean

file_regexs = ['r3.5', 'r8ts', 'r16ts', 'r24ts', 'r32ts', 'r46ts', 'rk1-16']

means = []
vars = []
for reg in file_regexs:
    m, v = get_mean_and_var(reg)
    means.append(m)
    vars.append(v)

means = np.array(means)
vars = np.array(vars)

all_means = means
all_vars = vars

means = means[1:]
vars = vars[1:]

z, cov = np.polyfit(means, vars, 1, cov=True)
print('Parameters:', z)
print('Fit errors:', np.sqrt(np.diag(cov)))

a = z[0]
b = z[1]
plt.scatter(means, vars, label='Wyniki pomiarów', s=29, c = 'b')
plt.scatter([all_means[0]], [all_vars[0]], s = 29, c = 'r', label = 'Odrzucony pomiar')
plt.plot(np.arange(-500, 60000, 7500), a*np.arange(-500, 60000, 7500) + b, 'g', label='Dopasowana prosta')
plt.xlim((0, 60000))
plt.legend()
plt.show()

