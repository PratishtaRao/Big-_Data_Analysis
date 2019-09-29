%Author: Pratishta Prakash Rao

%Finding values for x
x = 1:1:80;
%Finding values of y as a function of x
y = 20./x.* 60
%Plotting the values
plot(x,y)
title('Plot of time taken to go to work as a function of speed')
xlabel('Speed in mph')
ylabel('Time taken in minutes')