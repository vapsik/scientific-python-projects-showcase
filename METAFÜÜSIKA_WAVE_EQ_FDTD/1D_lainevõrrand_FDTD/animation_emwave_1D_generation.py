import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 1000 #dx-ide arv
m = 1
x = np.linspace(0, m, N+1)
#y = np.linspace(0, m, N+1)
#xv, yv = np.meshgrid(x, y)
c = 3e8 # valguse kiirus
dx = x[1] - x[0]
x_c = N//2 #laineallika koordinaat

dt = dx/c # stabiilsuse nimel peaks dt olema samas suurusjärgus dx/c-ga

Eyt0 = np.zeros_like(x)
Eytm = Eyt0.copy()
Eytm_prev = Eyt0.copy()
t = 0

D2 = (-2*np.diag(np.ones(N+1)) + np.diag(np.ones(N), 1) + np.diag(np.ones(N), -1))* ((1/dx)**2) # Laplace'i operaatori maatriks

#hetkel f(t) töötab vahemikus t = Ndt, kus N on vahemikus (0, 2000) ja sealt edasi jääb laine võnkuma vastava varasemalt saadud energiaga
T_generation = 2000*dt

def iteratsioon(Em, Em_prev, t):
    #returnib Em+1 ja Em 
    #kasutus: Em, Em-1 = Em+1, Em
    t_new = t + dt
    f = np.zeros_like(Em)
    if t_new <= T_generation:
        # moduleeritud Gaussi pulss
        A = 1.0  # amplituud
        L = m  # ruumipiirkonna pikkus
        f_min = c / L  # miinimum sagedus, mis vastab ühele perioodile ruumipiirkonnas
        f_max = c / (0.1 * L)  # maksimum sagedus, mis vastab kümnele perioodile ruumipiirkonnas
        f_0 = (f_min + f_max) / 2  # tsentraalne sagedus
        sigma = (2 * np.pi * L) / (9 * c)  # pulsi laius tuleneb seosest Δf ∝ 1/σ, 
        t_0 = T_generation * dt / 2  # pulsi ajaline tsenter
        # moduleeritud Gaussi pulsi välja-arvutamine
        f[x_c] = dx* A * np.exp(-((t_new - t_0)**2) / (2 * sigma**2)) * np.cos(2 * np.pi * f_0 * t_new)
    Em_next = 2*Em - Em_prev + (c**2 * dt**2) * (D2 @ Em) + f
    Em_next[0] = Em_next[-1] = 0
    return Em_next, Em, t_new

'''
#snapshot 10000
iterN =10000
for i in range(iterN):
    Eytm, Eytm_prev, t = iteratsioon(Eytm, Eytm_prev, t)


plt.plot(x[:], Eytm[:])
plt.title(f"iteratsioon = {iterN}")
plt.show()
A = 1

iterN = 2000
for i in range(iterN):
    Eytm, Eytm_prev, t = iteratsioon(Eytm, Eytm_prev, t)
    A = min(Eytm.max(), A)

plt.plot(x[:], Eytm[:])
plt.title(f"iteratsioon = {iterN}")
plt.show()
'''


# animatsiooni ülesseadmine
# NB! animatsiooni kiirust ja kaadrite tihedust iteratsioonisammude suhtes saab muuta muutes:
# parameetreid skip ja interval
A = 0.05
fig, ax = plt.subplots()
line_wave, = ax.plot(x, Eytm)
ax.set_ylim(-1.5 * A, 1.5 * A)
ax.set_xlabel("Koordinaat (x)")
ax.set_ylabel("Ristuv elektriväli (Ey)")

# iteratsiooni numbrile viitav tekst
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
t_steps = 10000 # animatsioon teeb täpselt 10 000 sammu
skip = 10
#initseerimine
Eytm = Eyt0.copy()
Eytm_prev = Eyt0.copy()
t = 0

#animeerimise ajal kogun andmed maatriksisse
# hetkel välja commentitud
# NB! kui töötavad, tekitavad suhteliselt suured failid (~!100MB)
'''
data_matrix1, data_matrix2 = [], [] 
data_matrix1.append(Eytm.copy())
'''

# iteratsiooni funktsioon animatsioonile, üles seatud Matplotlibi FuncAnimationi syntaxi tavaga vastavalt
def update(frame):
    global Eytm, Eytm_prev, t, A
    for _ in range(skip):  # skipib kaadrite vahel valitud arvu olekuid
        Eytm, Eytm_prev, t = iteratsioon(Eytm, Eytm_prev, t)
        '''
        # andmekogumine
        if t <= T_generation:
            data_matrix1.append(Eytm.copy())
        elif t <= t_steps * dt:
            data_matrix2.append(Eytm.copy()) # piiran andmemaatriksi suurust
        '''

    A = min(max(abs(Eytm)), A) if A > max(abs(Eytm)) else max(max(abs(Eytm)), A)
    line_wave.set_ydata(Eytm)
    ax.set_ylim(-1.5 * A, 1.5 * A) # sean graafikule uued y-piirid
    iteration_text.set_text(f'iteratsioon: {frame * skip}, t:{t}, t/dt: {t/dt}, f(t) rakendub: {t <= T_generation}, frame: {frame}')
    return line_wave, iteration_text

# animatsiooni loomine: kaadrite arv = 10000 sammu/10 sammu kaadri kohta; animatsioon jääb korduma
# animatsiooni kiirust saab muuta parameetrit interval muutes
ani = FuncAnimation(fig, update, frames=t_steps//skip, interval=50, blit=False)
# NB! animatsioon ei peatu, peale t_steps//skip kaadrit hakkab ta "uuesti" tööle
ani.save(filename="1Dlainevõrrand.mp4", writer="ffmpeg")

plt.grid()
# displayb animatsiooni
plt.show()

'''
# salvestan kogutud andmemaatriksid eraldi failidesse
data_matrix1 = np.array(data_matrix1)
data_matrix2 = np.array(data_matrix2)
np.savetxt("data_matrix_during_generation.csv", data_matrix1, delimiter=",")
np.savetxt("data_matrix_post_generation.csv", data_matrix2, delimiter=",")
'''