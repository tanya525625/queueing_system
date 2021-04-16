import random
import numpy as np
import matplotlib.pyplot as plot

lmd = 0.25
mu = 1 / 3
n = 3
minutes_for_model = 100 #Количество генерируемых заявок
p = np.zeros((minutes_for_model - n - 1, n + 1))
p_pred = np.zeros(n+1)
min_it = 10000
it_num = 500
h = 5 #Срез времени
reject_count = 0  # счетчик отказов

for i in range(it_num):
    requests = [] # время получения заявок
    last_requests_time = 0.0
    for minute in range(0, minutes_for_model - 1):
        rnd = random.expovariate(lmd) #Время между заявками
        last_requests_time += rnd
        requests.append(last_requests_time)
    if requests[-1] < min_it:
        min_it = requests[-1]
    # print(requests)

    handles = []
    # Для каждой заявки генерируем время ее обработки
    for request in requests:
        rnd = random.expovariate(mu)
        handles.append(rnd)
    # print(handles)


    handles_end = np.zeros(n) #вектор времени выхода заявки
    j = 0
    t = 0
    for i in range(n):
        handles_end[i] = requests[i] + handles[i]
    for request in range(n, len(requests)):
        k = 0 # счетчик занятых каналов
        for i in range(n):
            if handles_end[i] > requests[request]:
                k += 1
            p_pred[k] += 1

        c = 0  # счетчик занятых каналов
        if requests[request] > t:
            for i in range(n):
                if handles_end[i] > requests[request]:
                    c += 1
            p[j, c] += 1
            j += 1
            t += h

        ind = np.argmin(handles_end)
        if requests[request] < min(handles_end):
            reject_count += 1  # отказано в обслуживании
        else:
            handles_end[ind] = requests[request] + handles[request]


k = int(min_it/h)
p = p[:k,:]
time = np.arange(1,k,1)
p_1 = np.transpose(p)
time = np.insert(time,0,0)
for y in p_1.tolist():
    plot.plot(time, y)
plot.show()

for i in range(n+1):
    print('Предельная вероятность '+ str(i) + ': ' +  str(p_pred[i] / np.sum(p_pred)))

print('Вероятность отказа ' + str(reject_count/(minutes_for_model * it_num)))
print('Относительная пропускная способность ' + str(1-reject_count/(minutes_for_model * it_num)))
print('Абсолютная пропускная способность ' + str(lmd * (1-reject_count/(minutes_for_model * it_num))))
print('Среднее число занятых каналов ' + str((lmd * (1-reject_count/(minutes_for_model * it_num)))/mu))