function [classe1_treinamento,classe1_teste,classe1_validacao]= separador(classe1)
    classe1_treinamento = classe1(1:round(0.7*length(classe1)),:);
    num_teste = round(abs(round(0.7*length(classe1)) - length(classe1))*2/3);
    num_validacao = length(classe1)- (length(classe1_treinamento)+num_teste);
    
    classe1_teste = classe1(length(classe1_treinamento)+1:length(classe1_treinamento)+num_teste,:);
    classe1_validacao = classe1(length(classe1_teste)+1:length(classe1_teste)+num_validacao,:);
end