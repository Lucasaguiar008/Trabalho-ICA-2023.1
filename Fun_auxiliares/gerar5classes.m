
function [dados_treinamento,dados_teste,dados_validacao] = gerar5classes()

    addpath("Fun_auxiliares")
    addpath("Classes_separadas")
    
    classe1 = readtable('classe_1.txt');
    classe1 = table2array(classe1);
    
    classe2 = readtable('classe_2.txt');
    classe2 = table2array(classe2);
    
    classe3 = readtable('classe_3.txt');
    classe3 = table2array(classe3);
    
    classe4 = readtable('classe_4.txt');
    classe4 = table2array(classe4);
    
    classe5 = readtable('classe_5.txt');
    classe5 = table2array(classe5);
    
    % Definindo as quantidades
    qtd_atributos = 3;
    qtd_classes = 5;
    
    % Embaralhar
    classe1 = embaralhar(classe1);
    classe2 = embaralhar(classe2);
    classe3 = embaralhar(classe3);  
    classe4 = embaralhar(classe4); 
    classe5 = embaralhar(classe5); 
    
    [classe1_treinamento,classe1_teste,classe1_validacao]= separador(classe1);
    [classe2_treinamento,classe2_teste,classe2_validacao]= separador(classe2);
    [classe3_treinamento,classe3_teste,classe3_validacao]= separador(classe3);
    [classe4_treinamento,classe4_teste,classe4_validacao]= separador(classe4);
    [classe5_treinamento,classe5_teste,classe5_validacao]= separador(classe5);
    
    dados_treinamento = [classe1_treinamento;classe2_treinamento;classe3_treinamento;classe4_treinamento;classe5_treinamento];
    dados_teste = [classe1_teste;classe2_teste;classe3_teste;classe4_teste;classe5_teste];
    dados_validacao = [classe1_validacao;classe2_validacao;classe3_validacao;classe4_validacao;classe5_validacao];
    
    dados_treinamento = atualizar_classes(dados_treinamento,qtd_atributos,qtd_classes);
    dados_teste = atualizar_classes(dados_teste,qtd_atributos,qtd_classes);
    dados_validacao = atualizar_classes(dados_validacao,qtd_atributos,qtd_classes);
    
    dados_treinamento = embaralhar(dados_treinamento);
    dados_teste = embaralhar(dados_teste);
    dados_validacao = embaralhar(dados_validacao);

end
