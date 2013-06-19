function [] = plot_degree(x, sample, filename)
%Plot the degree log-log ccdf
%   Plot the ccdf of the in and out-degree of both x and sample.

% James Atwood 5/29/13
% jatwood@cs.umass.edu

[sx1,sx2] = size(x);

z = sqrt(sx2);

for i=1:sx1
  [deg(i), indeg(i), outdeg(i)] = degrees(reshape(x(i,:),z,z));
  [sdeg(i), sindeg(i), soutdeg(i)] = degrees(reshape(sample(i,:),z,z));
  
  [ixpdf(i),iypdf(i),ixcdf(i),iycdf(i),ilogk(i),ilogx(i)] = pdf_cdf_rank(indeg(i),'off');
  [isxpdf(i),isypdf(i),isxcdf(i),isycdf(i),islogk(i),islogx(i)] = pdf_cdf_rank(sindeg(i),'off');
  
  [oxpdf(i),oypdf(i),oxcdf(i),oycdf(i),ologk(i),ologx(i)] = pdf_cdf_rank(outdeg(i),'off');
  [osxpdf(i),osypdf(i),osxcdf(i),osycdf(i),oslogk(i),oslogx(i)] = pdf_cdf_rank(soutdeg(i),'off');
end

ix = log10(ixcdf);
iy = log10(1 - iycdf);
isx = log10(isxcdf);
isy = log10(1 - isycdf);

ox = log10(oxcdf);
oy = log10(1 - oycdf);
osx = log10(osxcdf);
osy = log10(1 - osycdf);

fig=figure;
%plot(ix(ix ~= -Inf & iy ~= -Inf), iy(ix ~= -Inf & iy ~= -Inf), 'bx-', isx(isx ~= -Inf & isy ~= -Inf), isy(isx ~= -Inf & isy ~= -Inf), 'cx-', ox(ox ~= -Inf & oy ~= -Inf), oy(ox ~= -Inf & oy ~= -Inf), 'rx-', osx(osx ~= -Inf & osy ~= -Inf), osy(osx ~= -Inf & osy ~= -Inf), 'mx-');
hold on;
errorbar(ix(ix ~= -Inf & iy ~= -Inf), iy(ix ~= -Inf & iy ~= -Inf), 'bx-')

title('Log-Log CCDF of Degree Distributions')
xlabel('Log10(degree)')
ylabel('Log10(ccdf)')

legend('Training In-Degree','Sample In-Degree','Training Out-Degree','Sample Out-Degree');

saveas(fig,filename,'pdf');

end

