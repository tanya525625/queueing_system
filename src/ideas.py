import random
import numpy as np

lmd = 0.25
n = 3
minutes_for_model = 60 #Количество генерируемых заявок

requests = [] # время получения заявок
last_requests_time = 0.0
for minute in range(0, minutes_for_model - 1):
    rnd = random.expovariate(lmd) #Время между заявками
    last_requests_time += rnd
    requests.append(last_requests_time)
print(requests)

handles = []
mu = 1 / 3
# Для каждой заявки генерируем время ее обработки
for request in requests:
    rnd = random.expovariate(mu)
    handles.append(rnd)
print(handles)

reject_count = 0 #счетчик отказов
handles_end = np.zeros(n) #вектор времени выхода заявки
p = np.zeros(( minutes_for_model - n -1, n+1))
j = 0

for i in range(n):
    handles_end[i] = requests[i] + handles[i]



for request in range(n, len(requests) ):
    c = 0 # счетчик свободных каналов
    for i in range(n):
         if handles_end[i] > requests[request]:
            c += 1
    p[j, c] += 1
    j +=1

    ind = np.argmin(handles_end)
    if requests[request] < min(handles_end):
        reject_count += 1  # отказано в обслуживании
    else:
        handles_end[ind] = requests[request] + handles[request]

print(p)