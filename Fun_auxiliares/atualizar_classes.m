function dados_pron = atualizar_classes(dados,qtd_atributos,qtd_classes)
    vetor = zeros(length(dados), qtd_classes);
    for i = 1:length(dados)
        vetor(i, dados(i,size(dados,2))) = 1;
        dados_pron(i,:) = [dados(i,1:qtd_atributos+1) vetor(i,:)];
    end
    dados_pron(:,4)=[];
end