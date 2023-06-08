function montartabela(vetor)
    vetor = vetor';
    valoresUnicos = unique(vetor);
    contagem = histcounts(vetor, [valoresUnicos, max(valoresUnicos)+1]);
    
    % Montar tabela com dados e contagem
    tabela = [valoresUnicos', contagem'];
    
    % Exibir tabela
    disp('Dados   | Quantidade');
    disp('-------------------');
    disp(tabela);
end
