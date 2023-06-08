clc; clear all; close all;
%% AUTORES
% LUCAS AGUIAR
% GEAN
% MICHELE
% ÚLTIMA EDIÇÃO: 30/05/2023

addpath("Fun_auxiliares")
addpath("Classes_separadas")
addpath("Dados_definidos")

matriz = [
    10;
    20;
    40;
    80;
    100;
    ];

for experimento=1:length(matriz)
    clc; close all;


    % Gerando as 5 classes
    gerar_novos_dados = false;
    if (gerar_novos_dados)
        [dados_treinamento,dados_teste,dados_validacao] = gerar5classes();
    else
        load('dados_treinamento_5classes.mat','dados_treinamento');
        load('dados_teste_5classes.mat','dados_teste');
        load('dados_validacao_5classes.mat','dados_validacao');
    end


    % Definindo as quantidades
    qtd_atributos = 3;
    qtd_classes = 5;

    dados_treinamento = [dados_treinamento;dados_validacao];

    %% INICIANDO A ELM

    taxa_acerto=[]; % variável para a taxa de acerto
    melhor_resultado = -Inf;

    for k=1:100
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
        H = matriz(experimento); % Número de neurônios na camada oculta

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
    matrizConfusao = matrizConfusao_ajuste(matrizConfusao);
    matrizConfusao(isnan(matrizConfusao)) = 0;
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



    %% Salvando a figura

    % Definir o novo nome do arquivo
    novoNome = ["ELM--N-"+num2str(H)+"_AcertoMax_"+ num2str(max(taxa_acerto),2)+".fig"];

    % Salvar a figura com o novo nome do arquivo
    savefig(novoNome);

end

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


