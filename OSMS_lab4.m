x = [1, 0, 1, 1, 0];
y = [1, 1, 1, 0, 1];

original = zeros(1, 31);
shifted = zeros(1, 31);

for i = 1:31
    original(i) = xor(x(5), y(5));
    shifted(i) = xor(x(5), y(5));

    sumx = xor(x(4), x(5));
    x(2:end) = x(1:end-1);
    x(1) = sumx;

    sumy = xor(y(2), y(5));
    y(2:end) = y(1:end-1);
    y(1) = sumy;
end

fprintf('Shift | Bits | AutoCorr\n');

corr = xcorr(original, 'coeff');

% Инициализация графика
figure;
plot(corr);

% Вывод графика автокорреляции
for shift = 0:30
    fprintf('%5d | ', shift+1);

    for i = 1:31
        fprintf('%d', shifted(i));
    end

    fprintf(' | ');

    autocorr_value = corr(abs(length(original) - shift) + 1);
    fprintf('%+1.3f\n', autocorr_value);

    shifted = [shifted(end), shifted(1:end-1)];
end
