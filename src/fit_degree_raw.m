function [fix, fiy, fox, foy] = fit_degree_raw(x)

[sx1,sx2] = size(x);
z = sqrt(sx2);

[deg, indeg, outdeg] = degrees(reshape(x,z,z));

[ixpdf,iypdf,ixcdf,iycdf,ilogk,ilogx] = pdf_cdf_rank(indeg,'off');

[oxpdf,oypdf,oxcdf,oycdf,ologk,ologx] = pdf_cdf_rank(outdeg,'off');

ix = log10(ixcdf);
iy = log10(1 - iycdf);

ox = log10(oxcdf);
oy = log10(1 - oycdf);

fix = ix(ix ~= -Inf & iy ~= -Inf);
fiy = iy(ix ~= -Inf & iy ~= -Inf);

fox = ox(ox ~= -Inf & oy ~= -Inf);
foy = oy(ox ~= -Inf & oy ~= -Inf);
