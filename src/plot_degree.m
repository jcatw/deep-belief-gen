function [] = plot_degree(x, sample, filename)
%Plot the degree log-log ccdf
%   Plot the ccdf of the in and out-degree of both x and sample.

% James Atwood 5/29/13
% jatwood@cs.umass.edu

[sx1,sx2] = size(x);

z = sqrt(sx2);

[deg, indeg, outdeg] = degrees(reshape(x,z,z));
[sdeg, sindeg, soutdeg] = degrees(reshape(sample,z,z));

[ixpdf,iypdf,ixcdf,iycdf,ilogk,ilogx] = pdf_cdf_rank(indeg,'off');
[isxpdf,isypdf,isxcdf,isycdf,islogk,islogx] = pdf_cdf_rank(sindeg,'off');

[oxpdf,oypdf,oxcdf,oycdf,ologk,ologx] = pdf_cdf_rank(outdeg,'off');
[osxpdf,osypdf,osxcdf,osycdf,oslogk,oslogx] = pdf_cdf_rank(soutdeg,'off');


ix = log10(ixcdf);
iy = log10(1 - iycdf);
isx = log10(isxcdf);
isy = log10(1 - isycdf);

ox = log10(oxcdf);
oy = log10(1 - oycdf);
osx = log10(osxcdf);
osy = log10(1 - osycdf);

size(osx)
size(osy)

fig=figure;
plot(ix(ix ~= -Inf & iy ~= -Inf), iy(ix ~= -Inf & iy ~= -Inf), 'bx-', isx(isx ~= -Inf & isy ~= -Inf), isy(isx ~= -Inf & isy ~= -Inf), 'cx-', ox(ox ~= -Inf & oy ~= -Inf), oy(ox ~= -Inf & oy ~= -Inf), 'rx-', osx(osx ~= -Inf & osy ~= -Inf), osy(osx ~= -Inf & osy ~= -Inf), 'mx-');

title('Log-Log CCDF of Degree Distributions')
xlabel('Log10(degree)')
ylabel('Log10(ccdf)')

legend('Training In-Degree','Sample In-Degree','Training Out-Degree','Sample Out-Degree');

saveas(fig,filename,'pdf');

end

