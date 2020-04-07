
R = 20;

outputVideoName = strcat('video_est',num2str(R),'.mp4');
% outputVideoName = strcat('video_org.mp4');
outputVideo = VideoWriter(outputVideoName,'MPEG-4');
outputVideo.FrameRate = 30;

open(outputVideo);
for i = 1:205
   img = imread(strcat('video_frame/video_frame',num2str(i),'_est20.jpg'));
   writeVideo(outputVideo,img);
end
close(outputVideo);