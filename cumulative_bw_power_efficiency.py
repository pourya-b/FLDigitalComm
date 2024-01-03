# for Tables 2 and 3
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

precision = 32
vip_length = 8
iter_num = 1000
I = 5  # aggregation interval in FL

comm_dQ_sc = 0  # for dynamic QAM
comm_4Q_sc = 0  # for 4-QAM
comm_perc = np.zeros(iter_num)  # cumulative communication bandwidth/latency

# Table 2. Dynamic changes QAM modulation in {16, 8, 4} sizes
for i in np.arange(0, iter_num, I):
    comm_4Q_sc += precision/2  # number of symbols tranmitted for each values. /2 is for 4-QAM where 2=log2(4)
    if i <= 300:
        comm_dQ_sc += vip_length/2 + (precision-vip_length)/4  # 4-QAM + 16-QAM
    elif i > 300 and i <= 450:
        comm_dQ_sc += vip_length/2 + (precision-vip_length)/3  # 4-QAM + 8-QAM
    else:
        comm_dQ_sc += precision/2  # 4-QAM + 4-QAM

    comm_perc[i] = comm_dQ_sc/comm_4Q_sc

iter_num = 1000
pow_ddb_sc = 0  # for dynamic power
pow_18db_sc = 0  # for fixed power of SNR = 18 dB
pow_perc = np.zeros(iter_num)  # cumulative transmit power

# Table 3. Dynamic changes tranmit power in SNR values {14, 16, 18} dB
for i in np.arange(0, iter_num, I):
    pow_18db_sc += 10 ** 1.8  # sig_power = 10^(SNR/10) * noise_power (noise powers get cancelled, so does not matter)
    if i <= 150:
        pow_ddb_sc += 10 ** 1.4
    elif i > 150 and i <= 600:
        pow_ddb_sc += 10 ** 1.6
    else:
        pow_ddb_sc += 10 ** 1.8

    pow_perc[i] = pow_ddb_sc/pow_18db_sc


# plotting
font_size = 30
fig, ax = plt.subplots(1, 1)
fig.set_figwidth(16)
fig.set_figheight(8)

ax.plot(100 * comm_perc.T, linewidth=5, color="cornflowerblue")

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.offsetText.set_fontsize(font_size)

plt.yticks(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.xlabel("iterations $t$", fontsize=font_size)
plt.title("communication bandwidth efficiency %", fontsize=font_size)
plt.grid(linestyle='dashed')
# plt.savefig("plot_comm_efficiency.eps")
plt.show()

font_size = 30
fig, ax = plt.subplots(1, 1)
fig.set_figwidth(16)
fig.set_figheight(8)

ax.plot(pow_perc.T, linewidth=5, color="cornflowerblue")

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.offsetText.set_fontsize(font_size)

plt.yticks(fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.xlabel("iterations $t$", fontsize=font_size)
plt.title("communication energy efficiency %", fontsize=font_size)
plt.grid(linestyle='dashed')
# plt.savefig("plot_comm_efficiency.eps")
plt.show()
