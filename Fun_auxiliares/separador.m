function [classe1_treinamento,classe1_teste,classe1_validacao]= separador(classe1)
    taxa_treinamento = 0.65;
    taxa_teste=0.2;
    
    classe1_treinamento = classe1(1:round(taxa_treinamento*length(classe1)),:);   

    num_teste = round(taxa_teste*length(classe1));
    num_validacao = length(classe1)- (length(classe1_treinamento)+num_teste);
    
    classe1_teste = classe1(length(classe1_treinamento)+1:length(classe1_treinamento)+num_teste,:);
    classe1_validacao = classe1(length(classe1_treinamento)+length(classe1_teste)+1:end,:);
end