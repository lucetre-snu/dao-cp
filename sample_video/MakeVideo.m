
R = 5;
for R = 5:5:20
    filename = strcat('./video_frame/CPALS', num2str(R));
    outputVideoName = strcat('Video_CPALS',num2str(R),'.mp4');
    % outputVideoName = strcat('Video_Org.mp4');
    outputVideo = VideoWriter(outputVideoName,'MPEG-4');
    outputVideo.FrameRate = 30;

    open(outputVideo);
    for i = 101:205
       img = imread(strcat(filename,'/video_frame',num2str(i),'.jpg'));
       writeVideo(outputVideo,img);
    end
end
close(outputVideo);