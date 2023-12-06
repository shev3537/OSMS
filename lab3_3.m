% Определение массивов a и b
a = [0.3 0.2 -0.1 4.2 -2 1.5 0];
b = [0.3 4 -2.2 1.6 0.1 0.1 0.2];

% Создание новой фигуры для графиков
figure;

% Построение графика a в первом подокне
subplot(3, 1, 1);
plot(a);
title('График a');
xlabel('Индекс элемента');
ylabel('Значение');

% Построение графика b во втором подокне
subplot(3, 1, 2);
plot(b);
title('График b');
xlabel('Индекс элемента');
ylabel('Значение');

format longG; % Установка формата вывода чисел

% Вычисление и вывод корреляции между a и b
ab = sum(a .* b);
fprintf('%.0f \n', ab)

% Нахождение максимального значения корреляции смещенных массивов
maxab = -9999;
nofmax = 0;

for i = 1:7
    b = circshift(b, 1);
    ab = sum(a .* b);
    
    % Вывод кореляции на каждой итерации
    fprintf('------------ %.0f ------------ \n', ab)
    
    % Поиск максимального значения корреляции и количества смещений
    if (ab > maxab)
        maxab = ab;
        nofmax = i;
    end
end

% Восстановление массива b с максимальным значением корреляции
b = circshift(b, nofmax);

% Построение восстановленного массива b в третьем подокне
subplot(3, 1, 3);
plot(b);
title('График b');
xlabel('Индекс элемента');
ylabel('Значение');
