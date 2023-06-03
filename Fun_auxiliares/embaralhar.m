function matriz_nova = embaralhar(dados)
    % Embaralhando os dados
    [l,~] = size(dados);
    index_aleatorios = randperm(l);
    matriz_nova = dados(index_aleatorios,:);
end