function [] = compare_degree(x, samples, n, filename)

fig = figure();

%FS = findall(fig,'-property','FontSize');
%set(FS,'FontSize',24);

%title('Comparison of Degree Distributions from Training and Generated Networks','FontSize',18);

title('Comparison of Network Degree Distributions','FontSize',18)

xlabel('log degree','FontSize',16);
ylabel('log ccdf','FontSize',16);

hold on
for i=1:n
  [fix, fiy, fox, foy] = fit_degree_raw(x(i,:));
  plot(fix,fiy,'gx-',fox,foy,'bx-');
  [fix, fiy, fox, foy] = fit_degree_raw(samples(i,:));
  plot(fix,fiy,'rx-',fox,foy,'kx-');
end

leg = legend('Krapivsky-Generated Networks: In-Degree','Krapivsky-Generated Networks: Out-Degree','DBN-Generated Networks: In-Degree','DBN-Generated Networks: Out-Degree');
set(leg,"fontsize",10);

text(1.75,-1.0,"\\lambda=1.0\n\\mu=1.0",'FontSize',14);

%legend('boxon');
%legend('right');
hold off

saveas(fig,filename,'png');
end
