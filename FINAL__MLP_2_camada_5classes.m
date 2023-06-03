clc; clear all; close all;
%% AUTORES
% LUCAS AGUIAR
% GEAN
% MICHELE
% ÚLTIMA EDIÇÃO: 01/06/2023

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


%% Dividindo os dados em treinamento teste e validação
X_treinamento = dados_treinamento(:, 1:qtd_atributos)';
Y_treinamento = dados_treinamento(:, qtd_atributos+1:end)';

X_validacao = dados_validacao(:, 1:qtd_atributos)';
Y_validacao = dados_validacao(:, qtd_atributos+1:end)';

X_teste = dados_teste(:, 1:qtd_atributos)';
Y_teste = dados_teste(:, qtd_atributos+1:end)';


%% AJUSTANDO OS DADOS PARA SEREM UTILIZADOS NA MLP
% Definir número máximo de épocas de treinamento
num_epocas = 1000;

% Definindo as configurações da rede neural
taxa_aprendizagem = 0.1;
num_entradas = 3;
num_oculta1 = 10;
num_oculta2 = 10;
num_saidas = qtd_classes;

% Inicializando os pesos aleatoriamente
pesos_oculta1 = rand(num_oculta1, num_entradas) - 0.5;
pesos_oculta2 = rand(num_oculta2, num_oculta1) - 0.5;
pesos_saida = rand(num_saidas, num_oculta2) - 0.5;

% Inicializando os bias
bias_oculta1 = -1;
bias_oculta2 = -1;
bias_saida = -1;

% Definindo a função de ativação
%funcao_ativacao = @(x) 1./(1 + exp(-x)); tipo=1; %sigmoide
funcao_ativacao = @(x) (1-exp(-x))./(1+exp(-x)); tipo=2; % Tangente Hiperólica
%funcao_ativacao = @(x) relu(x);  derivada_relu = @(x) 1*relu_derivada(x); tipo=3; % ReLu e sua Derivada

% Vetor para armazenar o erro por época de treinamento, validação e teste
erro_epoca_treinamento = zeros(1, num_epocas);
erro_epoca_validacao = zeros(1, num_epocas);
erro_epoca_teste = zeros(1, num_epocas);

% Contador para controlar o aumento consecutivo do erro de validação
contador_erro = 0;
melhor_erro_validacao = Inf; % Melhor erro de validação (inicializado como infinito)

% Treinamento da rede neural
for epoca = 1:num_epocas
    clc; disp(['Época: ' num2str(epoca)]);

    % Treinamento
    for i = 1:length(X_treinamento)
        d = Y_treinamento(:, i); % saída desejada
        entrada = X_treinamento(:, i); % entrada

        % Forward pass
        saida_oculta1 = funcao_ativacao(pesos_oculta1 * entrada + bias_oculta1);
        saida_oculta2 = funcao_ativacao(pesos_oculta2 * saida_oculta1 + bias_oculta2);
        saida = funcao_ativacao(pesos_saida * saida_oculta2 + bias_saida);

        % Calculando erro na saída
        erro = d - saida;

        switch tipo
            case 1
                % Backpropagation usando a sigmoide
                delta_saida = erro .* saida .* (1 - saida);
                delta_oculta2 = (pesos_saida' * delta_saida) .* saida_oculta2 .* (1 - saida_oculta2);
                delta_oculta1 = (pesos_oculta2' * delta_oculta2) .* saida_oculta1 .* (1 - saida_oculta1);

            case 2
                % Backpropagation usando a Tangente Hiperbólica
                delta_saida = erro .* 0.5 .* (1 - saida.^2);
                delta_oculta2 = (pesos_saida' * delta_saida) .* 0.5 .* (1 - saida_oculta2.^2);
                delta_oculta1 = (pesos_oculta2' * delta_oculta2) .* 0.5 .* (1 - saida_oculta1.^2);

            case 3
                % Backpropagation usando a ReLu
                delta_saida = erro .* derivada_relu(saida);
                delta_oculta2 = (pesos_saida' * delta_saida) .* derivada_relu(saida_oculta2);
                delta_oculta1 = (pesos_oculta2' * delta_oculta2) .* derivada_relu(saida_oculta1);

            otherwise
                disp('A função de ativação não foi escolhida.')
                break;
        end

        % Atualizando pesos
        pesos_saida = pesos_saida + taxa_aprendizagem * delta_saida * saida_oculta2';
        pesos_oculta2 = pesos_oculta2 + taxa_aprendizagem * delta_oculta2 * saida_oculta1';
        pesos_oculta1 = pesos_oculta1 + taxa_aprendizagem * delta_oculta1 * entrada';

        % Atualizando biases
        bias_saida = bias_saida + taxa_aprendizagem * delta_saida;
        bias_oculta2 = bias_oculta2 + taxa_aprendizagem * delta_oculta2;
        bias_oculta1 = bias_oculta1 + taxa_aprendizagem * delta_oculta1;

        % Atualizando o vetor de erro de treinamento
        erro_epoca_treinamento(epoca) = erro_epoca_treinamento(epoca) + mean(abs(erro));
    end

    % Validação
    erro_validacao = 0;
    for j = 1:length(X_validacao)
        d_validacao = Y_validacao(:, j); % saída desejada para validação
        entrada_validacao = X_validacao(:, j); % entrada de validação

        % Forward pass na validação
        saida_oculta1_validacao = funcao_ativacao(pesos_oculta1 * entrada_validacao + bias_oculta1);
        saida_oculta2_validacao = funcao_ativacao(pesos_oculta2 * saida_oculta1_validacao + bias_oculta2);
        saida_validacao = funcao_ativacao(pesos_saida * saida_oculta2_validacao + bias_saida);

        % Calculando erro na saída de validação
        erro_validacao = erro_validacao + mean(abs(d_validacao - saida_validacao));
    end

    % Atualizando vetor de erro de validação
    erro_epoca_validacao(epoca) = erro_validacao;


    % Teste
    erro_teste = 0;
    for k = 1:length(X_teste)
        d_teste = Y_teste(:, k); % saída desejada para teste
        entrada_teste = X_teste(:, k); % entrada de teste

        % Forward pass no teste
        saida_oculta1_teste = funcao_ativacao(pesos_oculta1 * entrada_teste + bias_oculta1);
        saida_oculta2_teste = funcao_ativacao(pesos_oculta2 * saida_oculta1_teste + bias_oculta2);
        saida_teste = funcao_ativacao(pesos_saida * saida_oculta2_teste + bias_saida);

        % Calcular erro na saída do teste
        erro_teste = erro_teste + mean(abs(d_teste - saida_teste));
    end

    % Atualizando vetor de erro de validação
    erro_epoca_teste(epoca) = erro_teste;


    % Verificando critério de parada (aumento do erro de validação)
    if epoca > 1 && erro_validacao > melhor_erro_validacao
        contador_erro = contador_erro + 1;
    else
        contador_erro = 0;
        melhor_erro_validacao = erro_validacao;
    end

    % Verificar critério de parada (X vezes consecutivas aumento do erro)
    if contador_erro >= 10000
        disp('Critério de parada atingido: erro de validação aumentou por 1000 vezes consecutivas.');
        break;
    end
end

% Truncar os vetores de erro de treinamento e validação
erro_epoca_treinamento = erro_epoca_treinamento(1:epoca);
erro_epoca_validacao = erro_epoca_validacao(1:epoca);
erro_epoca_teste = erro_epoca_teste(1:epoca);

% Plot do gráfico de erro por época de treinamento e validação
subplot(1, 2, 1);
hold on;
plot(1:epoca, erro_epoca_treinamento, 'b', 'LineWidth', 1.5);
plot(1:epoca, erro_epoca_validacao, 'r', 'LineWidth', 1.5);
plot(1:epoca, erro_epoca_teste, 'g', 'LineWidth', 1.5);

xlabel('Época');
ylabel('Erro');
title('Gráfico de Erro por Época');
legend('Treinamento', 'Validação','Teste');

% Teste
resultado = zeros(length(Y_teste), 3);
alturas_certas = [];
alturas_erradas = [];

for k = 1:length(X_teste)
    d_teste = Y_teste(:, k); % saída desejada para teste
    entrada_teste = X_teste(:, k); % entrada de teste

    % Forward pass no teste
    saida_oculta1_teste = funcao_ativacao(pesos_oculta1 * entrada_teste + bias_oculta1);
    saida_oculta2_teste = funcao_ativacao(pesos_oculta2 * saida_oculta1_teste + bias_oculta2);
    saida_teste = funcao_ativacao(pesos_saida * saida_oculta2_teste + bias_saida);

    [~, resultado(k, 1)] = max(d_teste); % desejado
    [~, resultado(k, 2)] = max(saida_teste); % obtido
    if resultado(k, 1) == resultado(k, 2)
        resultado(k, 3) = 1;
        alturas_certas = [entrada_teste(2);alturas_certas];
    else
        alturas_erradas = [entrada_teste(2);alturas_erradas];
    end

end

taxa_acerto = sum(resultado(:, 3))/length(resultado);
acuracia_percentual = taxa_acerto * 100;

disp(['Acurácia: ' num2str(acuracia_percentual) '%']);

% Gerar matriz de confusão
matrizConfusao = confusionmat(resultado(:, 2)', resultado(:, 1)');
matrizConfusao = matrizConfusao_ajuste(matrizConfusao);
matrizConfusao(isnan(matrizConfusao)) = 0;

subplot(1, 2, 2)
confusionchart(matrizConfusao)
title('Matriz de confusão - Teste');



%% Gerando a tabela de erros e acertos das alturas
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
%
%
%
% vetor_todos=unique([alturas_certas;alturas_erradas]);
%
% for i=1:length(vetor_todos)
%     cont=0;
%     for k=1:length(alturas_certas)
%         if vetor_todos(i,1)==alturas_certas(k)
%             cont = cont+1;
%         end
%         vetor_todos(i,2)=cont;
%     end
%
%     cont=0;
%     for k=1:length(alturas_erradas)
%         if vetor_todos(i,1)==alturas_erradas(k)
%             cont = cont+1;
%         end
%         vetor_todos(i,3)=cont;
%     end
%
%     vetor_todos(i,4)=vetor_todos(i,2)/(vetor_todos(i,2)+vetor_todos(i,3))*100;
% end


% Exibir tabela
% disp(' ');
% disp('Dados   | Porcentagem');
% disp('-------------------');
% disp([vetor_todos(:,1) vetor_todos(:,4)])


%% Salvando a figura
%
% % Definir o novo nome do arquivo
% novoNome = ["Camada_n1_"+num2str(num_oculta1)+"_Acerto_"+ num2str(acuracia_percentual,2)+".fig"];
%
% % Salvar a figura com o novo nome do arquivo
% savefig(novoNome);
%

