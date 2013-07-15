function [dbn] = dbninitgen(dbn)

L = dbn.L;	
% set generative parameters
dbn.gen.bias = cell(1,L+1);
dbn.gen.pair = cell(1,L);
for i=1:L
    dbn.gen.bias{i} = dbn.rec.bias{i};
    dbn.gen.pair{i} = dbn.rec.pair{i}';
end
dbn.gen.bias(L+1) = dbn.rec.bias(L+1);

dbn.gen.label.bias = dbn.rec.label.bias;
dbn.gen.label.pair = dbn.rec.label.pair';
