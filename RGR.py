import matplotlib.pyplot as plt
import numpy as np


def crc_gen(bit_arr):
    """
    Генерирует CRC (Cyclic Redundancy Check) для переданного массива битовой последовательности.
    Parameters:
    bit_arr: Массив битовой последовательности.
    Returns:
    Последние 7 бит CRC.
    """
    G = [1, 1, 1, 1, 0, 0, 0, 0]
    # Добавляем 7 нулей в конец
    bit_arr = np.concatenate((bit_arr, np.zeros(7, dtype=int)))
    
    for i in range(len(bit_arr)-8):
        if bit_arr[i] == 1:
            bit_arr[i:i+8] ^= G
    
    return bit_arr[-7:]

def crc_rec(bit_arr):
    """
    Рекурсивно вычисляет CRC (Cyclic Redundancy Check) для переданного массива битовой последовательности.
    Parameters:
    bit_arr: Массив битовой последовательности.
    Returns:
    Последние 7 бит CRC.
    """
    G = [1, 1, 1, 1, 0, 0, 0, 0]  # Порождающий полином G
    bit_arr = np.array(bit_arr, dtype=int)
    for i in range(len(bit_arr)-8):
        if bit_arr[i] == 1:
            bit_arr[i:i+8] ^= G
    
    return bit_arr[-7:]

def gold_gen(G):
    """
    Генерирует последовательность синхронизации.
    Parameters:
    G: Длина последовательности.
    Returns:
    np.array: Последовательность синхронизации.
    """
    x = [1, 0, 1, 1, 0]
    y = [1, 1, 1, 0, 1]
    gold_sequence = []
    
    for i in range(G):
        gold_sequence.append(x[4] ^ y[4])

        temp = x[3] ^ x[4]
        x = [temp] + x[:4]

        temp = y[1] ^ y[4]
        y = [temp] + y[:4]

    return np.array(gold_sequence)

def cor_reс(bit_arr, gold):
    """
    Выполняет корреляцию между переданным массивом битовой последовательности и последовательностью синхронизации.
    Parameters:
    bit_arr: Массив битовой последовательности.
    gold: Последовательность синхронизации.
    Returns:
    Позиции начал синхронизаций.
    """
    gold = np.repeat(gold, 5)
    correlation = np.correlate(bit_arr, gold, "valid")
    correlation /= len(gold)/1.5 # чтобы значения были от 0 до 1
    
    plt.figure(6)
    plt.title('График корреляции')
    plt.xlabel("Биты")
    plt.ylabel("Значение корреляции")
    plt.plot(correlation)
    #plt.show()
    index = []
    for i in range(len(correlation)):
        if correlation[i] > 0.90:
           index.append(i)       
    if len(index) == 2:
        return index
    else:
        return -1



def decrypt(bit_arr):
    """
    Декодирует массив значений выше или ниже порога.
    Parameters:
    bit_arr: Массив битовой последовательности.
    Returns:
    Расшифрованный массив.
    """
    decod = []
    for i in range(0, len(bit_arr), 5):
        sr_arr = np.mean(bit_arr[i:i+5])
        if sr_arr > 0.5:
            decod.append(1)
        else:
            decod.append(0)
    return decod


### 1 задание
# Ввод имени и фамилии
f_name = input("Введите имя: ")
l_name = input("Введите фамилию: ")

# Преобразование имени и фамилии в битовую строку ASCII-кодов
bit_sequence = []
for char in f_name + ' ' +  l_name:
    ascii_code = ord(char)  # Получаем ASCII-код символа
    binary_representation = format(ascii_code, '08b')  # Преобразуем ASCII-код в битовую строку длиной 8 символов
    bit_sequence.extend([int(bit) for bit in binary_representation])

### 2 задание
plt.figure(1)
plt.title('Битовая последовательность')
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(bit_sequence)
#plt.show()

# Преобразование массива в строку без пробелов и запятых
bit_string = ''.join(map(str, bit_sequence))
#print(f_name + ' ' +  l_name)
print('Бит. послед.',bit_string)

### 3 задание
crc = crc_gen(bit_sequence)
print('CRC', crc)

### 4 задание 
gold_arr = gold_gen(31)
print('gold_arr', gold_arr)

data = np.concatenate((gold_arr, bit_sequence, crc, gold_arr))

plt.figure(2)
plt.title('Данные + CRC + синхронизации')
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(data)
#plt.show()

### 5 задание
data_x = np.repeat(data, 5) # 5 отчётов на 1 бит
plt.figure(3)
plt.title('Данные + CRC + синхронизации(амплитудная модуляция)')
plt.xlabel("Временные отсчеты")
plt.ylabel("Значение")
plt.plot(data_x)
#plt.show()

### 6 задание
Nx_x2 = np.zeros(len(data_x)*2)
len_data = len(data_x)
print("Введите число начала передачи сигнала ( 0 -", len_data,")",end=' ')
start_sig = input()
start_sig = int(start_sig)
Nx_x2[start_sig:start_sig + len(data_x)] = data_x
plt.figure(4)
plt.title('Полученный сигнал')
plt.xlabel("Временные отсчеты")
plt.ylabel("Амплитуда")
plt.plot(Nx_x2)
#plt.show()

### 7 задание
noise = np.random.normal(0, 0.1, len(Nx_x2))
sig_noise = noise + Nx_x2
sig_noise2 = noise + Nx_x2
plt.figure(5)
plt.title('Полученный сигнал + шум')
plt.xlabel("Временные отсчеты")
plt.ylabel("Амплитуда")
plt.plot(sig_noise)
#plt.show()

### 8 задание
cor = cor_reс(sig_noise, gold_arr)
print('Начало и конец сигнала', cor)
sig_noise = sig_noise[cor[0]:cor[1]] # обрезаем от начала 1 синхронизации до начала 2
# plt.figure(7)
# plt.title('Полученный сигнал начиная с синхронизации')
# plt.plot(sig_noise)
#plt.show()

### 9 задание
dec = decrypt(sig_noise)
print('Полученные данные', dec)
plt.figure(7)
plt.title('Полученный сигнал начиная с синхронизации')
plt.xlabel("Время")
plt.ylabel("Амплитуда")
plt.plot(sig_noise)
#plt.show()

### 10 задание
dec = dec[len(gold_arr):]
print('Полученный данные без синхронизации', dec)

### 11 задание
crc_sig = crc_rec(dec)
print('Проверка crc',crc_sig)
err = 0
for i in range(len(crc_sig)):
    if crc_sig[i] == 1:
        err = 1
if err == 0:
    print('Предача прошла без ошибок')
else:
    print('БЫЛИ ошибки!!')

### 12 задание
dec = dec[:-7]
#print('len dec',len(dec))
# Декодирование битовой последовательности в строку ASCII
bit_string = ''.join(str(b) for b in dec)
n = int(bit_string, 2)
decoded_string = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

print(decoded_string)
### 13 задание
fft_tx = abs(np.fft.fft(Nx_x2)) + 30
fft_rx = abs(np.fft.fft(sig_noise2))

fft_tx = np.fft.fftshift(fft_tx)
fft_rx = np.fft.fftshift(fft_rx)
plt.figure(8)
plt.title('Спектры полученного и переданного сигнала')
plt.xlabel("Частота [Гц]")
plt.ylabel("Амплитуда")
plt.plot(fft_tx) ## Без шума
plt.plot(fft_rx) ## С шумом
#plt.show()

data_05x = np.repeat(data, 3)
data_1x = np.repeat(data, 5)
data_2x = np.repeat(data, 10)

data_1x = data_1x[:len(data_05x)]
data_2x = data_2x[:len(data_05x)]

data_05x = abs(np.fft.fft(data_05x))+80
data_1x = abs(np.fft.fft(data_1x))+40
data_2x = abs(np.fft.fft(data_2x))

data_05x = np.fft.fftshift(data_05x)
data_1x = np.fft.fftshift(data_1x)
data_2x = np.fft.fftshift(data_2x)

plt.figure(9)
plt.title('Спектры 3 разных по длительности сигналов')
plt.xlabel("Частота [Гц]")
plt.ylabel("Амплитуда")
plt.plot(data_05x)
plt.plot(data_1x)
plt.plot(data_2x)
plt.show()