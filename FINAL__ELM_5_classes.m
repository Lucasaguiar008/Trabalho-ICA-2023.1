clc; clear all; close all;
%% AUTORES
% LUCAS AGUIAR
% GEAN
% MICHELE
% ÚLTIMA EDIÇÃO: 30/05/2023

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

dados_treinamento = [dados_treinamento;dados_validacao];

%% INICIANDO A ELM

taxa_acerto=[]; % variável para a taxa de acerto
melhor_resultado = -Inf;

for k=1:10
    clc; disp("Experimento: "+ num2str(k))

    X = dados_treinamento(:,1:qtd_atributos)';
    Y = dados_treinamento(:, qtd_atributos+1:end)';
    X_test = dados_teste(:,1:qtd_atributos)';
    Y_test = dados_teste(:, qtd_atributos+1:end)';

    % Variáveis auxiliares para verificar a influência de H
    alturas_erradas=[];
    alturas_certas=[];

    % Dados para a matriz de confusão
    verdadeiro = []; % variável que receberá os dados verdadeiros
    previsto = [];% variável que receberá os dados previstos

    % Parâmetros da ELM
    H = 60; % Número de neurônios na camada oculta

    % Inicialização aleatória dos pesos e bias
    W = randn(H, size(X,1)); % Matriz de pesos de dimensão H x n
    b = randn(H, 1); % Vetor de bias de dimensão H x 1

    % Função de ativação
    %g = @(x) (1 ./ (1 + exp(-x))); % Função sigmoide
    g = @(x) (1-exp(-x))./(1+exp(-x));% Tangente Hiperólica
    %g = @(x) relu(x); % ReLu e sua Derivada

    % Cálculo da saída da camada oculta
    G = g(W * X + b * ones(1, size(X, 2))); % Matriz de saída da camada oculta

    % Cálculo dos pesos da camada de saída utilizando regressão linear
    %beta = Y * pinv(G); % Método pinv
    beta = Y*G'*(G*G')^-1; % Método Jarbas

    % Testes
    Y_pred = beta * g(W * X_test + b * ones(1, size(X_test, 2))); % Saída predita para os dados de teste

    [~,Desejado] = max(Y_test);
    verdadeiro = [Desejado;verdadeiro];

    [~,Obtido] = max(Y_pred);
    previsto = [Obtido;previsto];

    cont = 0;
    for i=1:length(Desejado)
        if Desejado(i)==Obtido(i)
            cont=cont+1;
            alturas_certas = [X_test(2,i);alturas_certas];
        else
            alturas_erradas = [X_test(2,i);alturas_erradas];
        end
    end

    taxa_acerto=[taxa_acerto;cont*100/length(Desejado)];

    % Salva os dados do melhor resultado
    if (cont*100/length(Desejado))>melhor_resultado
        melhor_resultado = (cont*100/length(Desejado)); % melhortaxa de acerto
        matrizConfusao = confusionmat(verdadeiro, previsto); % Gerar matriz de confusão do melhor resultado
        M_alturas_certas = alturas_certas;
        M_alturas_erradas = alturas_erradas;
    end

end


%% PLOT Resultados

% Exibir a matriz de confusão
subplot(1,2,2)
confusionchart(matrizConfusao)
disp("Maior taxa de acerto: " + num2str(melhor_resultado) + "%")
title('Matriz de confusão do melhor experimento')

subplot(1,2,1)

% Título geral
sgtitle("Experimento com "+num2str(H)+" neurônios.")

k=1:1:length(taxa_acerto);
hold on
title('Taxa de acerto por experimento')
%plot(k,taxa_acerto,'Color','k','LineStyle',':')
bar(taxa_acerto,'FaceColor', [0.5 0.5 0.5])

v_media = repmat(mean(taxa_acerto),1,length(taxa_acerto));
plot(k,v_media,'Color','b','LineWidth',2)

v_melhor = repmat(melhor_resultado,1,length(taxa_acerto));
plot(k,v_melhor,'Color','g','LineWidth',2)

v_min = repmat(min(taxa_acerto),1,length(taxa_acerto));
plot(k,v_min,'Color','r','LineWidth',2)

%legend('Taxa de acerto por experimento', 'Média','Maior taxa de acerto','Menor taxa de acerto','Location', 'southeast');
legend("Taxa de acerto por experimento", "Média: "+num2str(mean(taxa_acerto),4)+"%","Maior taxa de acerto: "+num2str(max(taxa_acerto),4)+"%","Menor taxa de acerto: "+num2str(min(taxa_acerto),4)+"%",'Location', 'southeast');



%     %% Salvando a figura
%
%     % Definir o novo nome do arquivo
%     novoNome = ["ELM--N-"+num2str(H)+"_AcertoMax_"+ num2str(max(taxa_acerto),2)+".fig"];
%
%     % Salvar a figura com o novo nome do arquivo
%     savefig(novoNome);
%
%
%
% %% Gerando a tabela de erros e acertos das alturas
%
% % Obtendo valores únicos e contagem de ocorrências
% disp(' ');
% disp('--------------------------------');
% disp('-------Altura certas-----------');
% montartabela(alturas_certas)
%
% disp(' ');
% disp('--------------------------------');
% disp('-------Altura Erradas-----------');
% montartabela(alturas_erradas)


