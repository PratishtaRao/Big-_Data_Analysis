%Author: Pratishta Prakash Rao
%Front gear A
a = 73
%Front gear B
b = 51
%Front gear C
c = 31
%Number of teeth on the back gear
cog =[19 23 33 41 53 63 71]
%Gear Ratio for A
gr1= a ./ cog
%Gear Ratio for B
gr2= b ./cog
%Gear Ration for C
gr3= c ./ cog
%Plotting the data points
plot(cog,gr1,'g')
hold on 
plot(cog, gr2, 'b')
hold on 
plot (cog, gr3, 'm')
hold off
title('Plot for all possible gear ratio combinations')
xlabel('Number of teeth on the back cog')
ylabel('Gear ratio')
