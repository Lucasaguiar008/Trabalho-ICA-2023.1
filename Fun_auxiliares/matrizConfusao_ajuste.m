function matrizConfusao = matrizConfusao_ajuste(matrizConfusao)
    
    %--------------------------------------------------------
    cont=1; %Contador
    el=1; %Elemento
    New_MC=zeros(length(matrizConfusao)+1,length(matrizConfusao)+1);
    acc=0;
    tt=0;
    while el<length(New_MC+1)
        %Sens
        ca=1; %Cont Aux
        aux=0; %Somador aux
    
        while ca<length(matrizConfusao)+1
            New_MC(el,ca) = matrizConfusao (el,ca);
            aux=aux+  matrizConfusao(el,ca);
            tt= tt +matrizConfusao(el,ca);
            ca=ca+1;
        end
        New_MC(el,length(New_MC)) = round(matrizConfusao (el,el)/aux*100);
    
        %Precs
    
        ca=1;
        aux=0; %Somador aux
    
        while ca<length(matrizConfusao)+1
            New_MC(ca,el) = matrizConfusao (ca,el);
            aux=aux+  matrizConfusao (ca,el);
            ca=ca+1;
        end
        New_MC(length(New_MC),el)=  round(matrizConfusao (el,el)/aux*100);
        acc=acc+matrizConfusao (el,el);
        el=el+1;
    end
    New_MC(length(matrizConfusao)+1,length(matrizConfusao)+1)=round(acc/tt*100);
    New_MC;
    matrizConfusao=New_MC;

end