import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy import interpolate
from scipy.optimize import minimize

my_data = 'STARNET/NET6.DAT'


def filter_val(filterdata, lamb):
    lambdas = filterdata[:, 0]
    vals = filterdata[:, 1]
    f = interpolate.interp1d(lambdas, vals, fill_value=0, bounds_error=False, kind='linear', assume_sorted=True)
    return f(lamb)


def load_filter_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    content = content[2:-1]
    content = np.array(content, np.float32)
    content = content[:, [1, 2]]
    return content


def spectra_val(stardata, lamb):
    lambdas = stardata[:, 0]
    vals = stardata[:, 1]
    f = interpolate.interp1d(lambdas, vals, fill_value=999, bounds_error=False, kind='linear', assume_sorted=True)
    return f(lamb)


def load_star_data(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    mag_line = content[1]
    mag_val = float(mag_line.split('=')[1].split()[0])
    content = content[3:-1]
    content = [x.split() for x in content]
    content = np.array(content, np.float32)
    return content, mag_val


v_filter_data = load_filter_data('STARNET/vfilter.pro')
cn_filter_data = load_filter_data('STARNET/cnfilter.pro')
c2_filter_data = load_filter_data('STARNET/c2filter.pro')

star_data3775, mag3775 = load_star_data('STARNET/BS3775.SPC')
star_data4368, mag4368 = load_star_data('STARNET/BS4368.SPC')
star_data4694, mag4694 = load_star_data('STARNET/BS4694.SPC')
star_data5351, mag5351 = load_star_data('STARNET/BS5351.SPC')

v_filter_func = lambda x: filter_val(v_filter_data, x)
cn_filter_func = lambda x: filter_val(cn_filter_data, x)
c2_filter_func = lambda x: filter_val(c2_filter_data, x)
flux3775_func = lambda x: 10**(-0.4 * spectra_val(star_data3775, x))
flux4368_func = lambda x: 10**(-0.4 * spectra_val(star_data4368, x))
flux4694_func = lambda x: 10**(-0.4 * spectra_val(star_data4694, x))
flux5351_func = lambda x: 10**(-0.4 * spectra_val(star_data5351, x))


def subint1(filter_func, flux_func):
    return lambda x: filter_func(x) * flux_func(x) / (x**2)


def subint2(filter_func):
    return lambda x: filter_func(x) / (x**2)


def integral(filter_func, flux_func, l1 = 200.0, l2 = 900.0, limit = 400):
    int1, err1 = integrate.quad(subint1(filter_func, flux_func), l1, l2, limit=limit)
    int2, err2 = integrate.quad(subint2(filter_func), l1, l2, limit=limit)
    summed = -2.5 * np.log10(int1) + 2.5 * np.log10(int2)
    return summed


C3775 = mag3775 - integral(v_filter_func, flux3775_func, l1=472.0, l2=740.0)
magC2_3775 = integral(c2_filter_func, flux3775_func, l1=500, l2=530) + C3775
magCN_3775 = integral(cn_filter_func, flux3775_func, l1=375, l2=400) + C3775
print('3775', mag3775, magC2_3775, magCN_3775, C3775)

C4368 = mag4368 - integral(v_filter_func, flux4368_func, l1=472.0, l2=740.0)
magC2_4368 = integral(c2_filter_func, flux4368_func, l1=500, l2=530) + C4368
magCN_4368 = integral(cn_filter_func, flux4368_func, l1=375, l2=400) + C4368
print('4368', mag4368, magC2_4368, magCN_4368, C4368)

C4694 = mag4694 - integral(v_filter_func, flux4694_func, l1=472.0, l2=740.0)
magC2_4694 = integral(c2_filter_func, flux4694_func, l1=500, l2=530) + C4694
magCN_4694 = integral(cn_filter_func, flux4694_func, l1=375, l2=400) + C4694
print('4694', mag4694, magC2_4694, magCN_4694, C4694)

C5351 = mag5351 - integral(v_filter_func, flux5351_func, l1=472.0, l2=740.0)
magC2_5351 = integral(c2_filter_func, flux5351_func, l1=500, l2=530) + C5351
magCN_5351 = integral(cn_filter_func, flux5351_func, l1=375, l2=400) + C5351
print('5351', mag5351, magC2_5351, magCN_5351, C5351)


C2diff1 = magC2_4368 - magC2_3775
C2diff2 = magC2_4694 - magC2_4368
C2diff3 = magC2_5351 - magC2_4694
C2diff4 = magC2_4694 - magC2_3775
C2diff5 = magC2_5351 - magC2_4368

CNdiff1 = magCN_4368 - magCN_3775
CNdiff2 = magCN_4694 - magCN_4368
CNdiff3 = magCN_5351 - magCN_4694
CNdiff4 = magCN_4694 - magCN_3775
CNdiff5 = magCN_5351 - magCN_4368

print(C2diff1, C2diff2, C2diff3, C2diff4, C2diff5)
print(CNdiff1, CNdiff2, CNdiff3, CNdiff4, CNdiff5)

# 4368-3775       0.847      1.218
# 4694-4368       1.865      1.724
# 5351-4694      -2.109     -2.001
# 4694-3775       2.641      2.871
# 5351-4368      -0.213     -0.301

C2mags_syn = np.array([magC2_3775, magC2_4368, magC2_4694, magC2_5351])
C2mag_diff_syn = np.array([C2diff1, C2diff2, C2diff3, C2diff4, C2diff5])
C2mag_diff_obs = np.array([1.218, 1.724, -2.001, 2.871, -0.301])
C2magsynsum = np.sum(C2mags_syn)

def func(x):
    diff1 = x[1]-x[0]
    diff2 = x[2]-x[1]
    diff3 = x[3]-x[2]
    diff4 = x[2]-x[0]
    diff5 = x[3]-x[1]
    diffs = [diff1, diff2, diff3, diff4, diff5]
    res = 0.0
    for i in range(5):
        z = (diffs[i] - C2mag_diff_obs[i])**2
        res += z
    return res

def func_deriv(x):
    """ Derivative of objective function """
    dfdx0 = -2*(x[1] - x[0] - C2mag_diff_obs[0]) - 2*(x[2] - x[0] - C2mag_diff_obs[3])
    dfdx1 = 2*(x[1] - x[0] - C2mag_diff_obs[0]) - 2*(x[2] - x[1] - C2mag_diff_obs[1]) - 2*(x[3] - x[1] - C2mag_diff_obs[4])
    dfdx2 = 2*(x[2] - x[1] - C2mag_diff_obs[1]) - 2*(x[3] - x[2] - C2mag_diff_obs[2]) + 2*(x[2] - x[0] - C2mag_diff_obs[3])
    dfdx3 = 2*(x[3] - x[2] - C2mag_diff_obs[2]) + 2*(x[3] - x[1] - C2mag_diff_obs[4])
    return np.array([ dfdx0, dfdx1, dfdx2, dfdx3 ])

cons = ({'type': 'eq',
    'fun' : lambda x: np.array([x[0] + x[1] + x[2] + x[3] - C2magsynsum]),
    'jac' : lambda x: np.array([1.0, 1.0, 1.0, 1.0])})

res = minimize(func, C2mags_syn, jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
C2mags_corr = res.x
print(C2mags_corr, C2mags_syn)

CNmags_syn = np.array([magCN_3775, magCN_4368, magCN_4694, magCN_5351])
CNmag_diff_syn = np.array([CNdiff1, CNdiff2, CNdiff3, CNdiff4, CNdiff5])
CNmag_diff_obs = np.array([0.847, 1.865, -2.109, 2.641, -0.213])
CNmagsynsum = np.sum(CNmags_syn)

def func(x):
    diff1 = x[1]-x[0]
    diff2 = x[2]-x[1]
    diff3 = x[3]-x[2]
    diff4 = x[2]-x[0]
    diff5 = x[3]-x[1]
    diffs = [diff1, diff2, diff3, diff4, diff5]
    res = 0.0
    for i in range(5):
        z = (diffs[i] - CNmag_diff_obs[i])**2
        res += z
    return res

def func_deriv(x):
    """ Derivative of objective function """
    dfdx0 = -2*(x[1] - x[0] - CNmag_diff_obs[0]) - 2*(x[2] - x[0] - CNmag_diff_obs[3])
    dfdx1 = 2*(x[1] - x[0] - CNmag_diff_obs[0]) - 2*(x[2] - x[1] - CNmag_diff_obs[1]) - 2*(x[3] - x[1] - CNmag_diff_obs[4])
    dfdx2 = 2*(x[2] - x[1] - CNmag_diff_obs[1]) - 2*(x[3] - x[2] - CNmag_diff_obs[2]) + 2*(x[2] - x[0] - CNmag_diff_obs[3])
    dfdx3 = 2*(x[3] - x[2] - CNmag_diff_obs[2]) + 2*(x[3] - x[1] - CNmag_diff_obs[4])
    return np.array([ dfdx0, dfdx1, dfdx2, dfdx3 ])

cons = ({'type': 'eq',
    'fun' : lambda x: np.array([x[0] + x[1] + x[2] + x[3] - CNmagsynsum]),
    'jac' : lambda x: np.array([1.0, 1.0, 1.0, 1.0])})

res = minimize(func, CNmags_syn, jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
CNmags_corr = res.x
print(CNmags_corr, CNmags_syn)







