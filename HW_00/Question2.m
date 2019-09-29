%Author:Pratishta Prakash Rao

%Values of speed with increments of x1
x1 = 1:1:80;
%Values of y as a function of x1
y1 = 20./x1.* 60
%Values of speed with increments of 5
x2 = 5:5:75
%Values of y as a function of x2
y2 = 20./x2.*60
%Plotting the data points
plot(x1,y1,'m')
hold on 
plot(x2,y2,'--bs')
hold off
title('Plot of time taken to go to work as a function of speed')
xlabel('Speed in mph')
ylabel('Time taken in minutes')