addpath('../tensorlab_2016-03-28');
warning('off', 'all');


% 205 * 180(31~210) * 320 * 3

currentPath = fileparts(mfilename('fullpath'));
videoTensor = [];
R = 30;
dims = [205 180 320 3];
numOfFrames = dims(1);
tao = 100;
% numOfFrames = 10;
% tao = 5;
iterFrame = 5;

N = numOfFrames / iterFrame;
% N = 1;
videoTensor = NaN(dims);

for i = 1:N
    tensorFile = fopen(strcat(currentPath, '/video', num2str(i-1), '.tensor'), 'r');
    disp(i);
    tic;

    X = fscanf(tensorFile, "%d %d %d %d %d", [5, inf]);
    for row = X
        if row(2) > 30 & row(2) <= 210
            videoTensor(row(1), row(2)-30, row(3), row(4)) = row(5);
        end
    end
    fclose(tensorFile);
    toc;
end


% T = videoTensor(:, :, :, :);
% whos T
% options.Display = true; % Show progress on the command line.
% options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
% options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
% options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
% options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
% options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
% options.Refinement = false;
% options.incomplete = false;
% Uest = cpd(T, R, options);
% whos Uest
% Test = cpdgen(Uest);
for R = 5:5:20
    filename = strcat('./video_frame/CPALS', num2str(R));
    mkdir(filename);
    outputVideoName = strcat(filename, '/video_est.mp4');
    % outputVideoName = strcat('video_org.mp4');
    outputVideo = VideoWriter(outputVideoName,'MPEG-4');
    outputVideo.FrameRate = 30;
    open(outputVideo);

    % for frame = 1:numOfFrames    
    %     img = uint8(squeeze(Test(frame, :, :, :)));
    %     writeVideo(outputVideo,img);
    %     imwrite(img, strcat(filename, '/video_frame', num2str(frame), '.jpg'));
    % end

    minibatchSize = 1;
    for t = 1:minibatchSize:numOfFrames-tao
        frame = tao+t;
        disp(frame);
        tic;
        T = videoTensor(1:frame, :, :, :);
        whos T
        options.Display = true; % Show progress on the command line.
        options.Initialization = @cpd_rnd; % Select pseudorandom initialization.
        options.Algorithm = @cpd_als; % Select ALS as the main algorithm.
        options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
        options.AlgorithmOptions.TolFun = 1e-12; % Set function tolerance stop criterion
        options.AlgorithmOptions.TolX   = 1e-12; % Set step size tolerance stop criterion
        options.Refinement = false;
        options.incomplete = false;
        Uest = cpd(T, R, options);
        % whos Uest
        Test = cpdgen(Uest);

        testRuntime(t) = toc;
        testNormErr(t) = frob(Test-T);
        testFitness(t) = 1-testNormErr(t)/frob(T);

        img = uint8(squeeze(Test(frame, :, :, :)));
        writeVideo(outputVideo,img);
        imwrite(img, strcat(filename, '/video_frame', num2str(frame), '.jpg'));
    end
    close(outputVideo);

    testRuntime_Fitness = [testRuntime', testFitness']
end