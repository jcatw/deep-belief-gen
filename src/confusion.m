function [confusion_matrix] = confusion(truth, predictions)

[N,nlab] = size(truth);

confusion_matrix = zeros(nlab,nlab);

for i=1:N
    confusion_matrix(find(truth(i,:),1), find(predictions(i,:),1)) = ...
      confusion_matrix(find(truth(i,:),1), find(predictions(i,:),1)) + 1;
end

confusion_matrix = confusion_matrix/N;

