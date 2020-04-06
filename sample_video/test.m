addpath('../tensorlab_2016-03-28');
warning('off', 'all');


% 205 * 180(31~210) * 320 * 3

currentPath = fileparts(mfilename('fullpath'));
videoTensor = [];
R = 5;
dims = [205 180 320 3];
numOfFrames = dims(1);
iterFrame = 5;
% N = numOfFrames / iterFrame;
N = 1;
videoTensor = NaN(dims);

for i = 1:N
    tensorFile = fopen(strcat(currentPath, '/video', num2str(i-1), '.tensor'), 'r');
    disp(i);
    tic;

    X = fscanf(tensorFile, "%d %d %d %d %d", [5, inf]);
    for row = X
        if row(2) > 30 & row(2) < 210
            videoTensor(row(1), row(2)-30, row(3), row(4)) = row(5);
        end
    end
    fclose(tensorFile);

    for j = 1:iterFrame
    % for j = 1:1
        frame = (i-1)*iterFrame + j;
        T = videoTensor(1:frame, :, :, :);
        whos T

        options.Display = false;
        options.Algorithm = @cpd_als;
        options.Refinement = false;
        % options.TolX = 1e-12;
        Uest = cpd(T, R, options);
        whos Uest
        Test = cpdgen(Uest);

        imwrite(uint8(squeeze(T(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_org.jpg'));
        imwrite(uint8(squeeze(Test(frame, :, :, :))), strcat('./video_frame/video_frame', num2str(frame), '_est.jpg'));

    end
end