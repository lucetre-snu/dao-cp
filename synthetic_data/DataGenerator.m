% addpath('../packages/tensorlab_2016-03-28');
% warning('off', 'all');

% % Syntheic data generation


% split_points = [0 100 200 250 500 600 700 750 800 950 1000];
% theme = [       1 1   2   2   2   3   4   5   5   5];
% split_size = zeros(1, size(split_points, 2) - 1);
% split_N = size(split_size, 2);

% num = 0;
% prevTheme = 0;
% for i = 1:split_N
%     split_size(i) = split_points(i+1) - split_points(i);

%     size_tens = [10 20 30]; R = 10;
%     if prevTheme ~= theme(i)
%         U0 = cpd_rnd(size_tens, R);
%         T0 = round(cpdgen(U0)*100);
%         T0 = reshape(T0, [1 size_tens]);
%         prevTheme = theme(i);
%     else
%         U1 = cpd_rnd(size_tens, R);
%         T1 = round(cpdgen(U1)*10);
%         T1 = reshape(T1, [1 size_tens]);
%         T0 = T0 + T1;
%     end
    
%     U = cpd_rnd([split_size(i) size_tens], R);

%     T = cpdgen(U);
%     whos T T0
%     T = T + T0;

%     X = reshape(T, [1 10*20*30*split_size(i)]);
%     for j = 1:split_size(i)/5
%         filename = strcat('data/data', num2str(num), '.tensor');
%         fp = fopen(filename, 'w');
%         num = num + 1;
%         for k = 1:5
%             t = split_points(i) + (j-1)*5+k;
%             for l = 1:size(T, 2)
%                 for m = 1:size(T, 3)
%                     for n = 1:size(T, 4)
%                         row = sprintf('%d\t%d\t%d\t%d\t%.2f\n', t, l, m, n, T(t-split_points(i), l, m, n));
%                         fprintf(fp, '%s', row);
%                     end
%                 end
%             end
%         end
%         fclose(fp);
%     end
% end


addpath('../packages/tensorlab_2016-03-28');
warning('off', 'all');

% Syntheic data generation


split_points = [0 100 200 250 500 600 700 750 800 950 1000];
theme = [       1 1   2   2   2   3   4   5   5   5];
split_size = zeros(1, size(split_points, 2) - 1);
split_N = size(split_size, 2);

num = 0;
prevTheme = 0;
for i = 1:split_N
    split_size(i) = split_points(i+1) - split_points(i);

    size_tens = [10 20 30];
    if prevTheme ~= theme(i)
        T0 = round(randn(size_tens)*100);
        T0 = reshape(T0, [1 size_tens]);
        prevTheme = theme(i);
    else
        T1 = round(randn(size_tens)*10);
        T1 = reshape(T1, [1 size_tens]);
        T0 = T0 + T1;
    end
    
    T = randn([split_size(i) size_tens]);
    whos T T0
    T = T + T0;

    X = reshape(T, [1 10*20*30*split_size(i)]);
    for j = 1:split_size(i)/5
        filename = strcat('data/data', num2str(num), '.tensor');
        fp = fopen(filename, 'w');
        num = num + 1;
        for k = 1:5
            t = split_points(i) + (j-1)*5+k;
            for l = 1:size(T, 2)
                for m = 1:size(T, 3)
                    for n = 1:size(T, 4)
                        row = sprintf('%d\t%d\t%d\t%d\t%.2f\n', t, l, m, n, T(t-split_points(i), l, m, n));
                        fprintf(fp, '%s', row);
                    end
                end
            end
        end
        fclose(fp);
    end
end
