function [Cost, F1, accuracy] = TestNN(w2,w3,w4,b2,b3,b4,X,Y)
%Calculamos los outputs de nuestra red
m=size(X,1);
Y_test=zeros(m,3);
fp1=0; fp2=0; fp3=0; tp1=0; tp2=0; tp3=0; fn1=0; fn2=0; fn3=0;
[h_theta]=FuncTheta(w2,w3,w4,b2,b3,b4,X);% Funcion que predice el valor 3 dimensiones en la que el valor mayor de probabilidad indica el objeto que corresponde
for i=1:size(Y,1)
    [~,I]=max(h_theta(:,i));
    Y_test(i,I)=1; 
    des=0;
    if Y_test(i,:)==[1 0 0]
        des=1;
    elseif Y_test(i,:)==[0 1 0]
        des=2;
    elseif Y_test(i,:)==[0 0 1]
        des=3;
    end
    switch des
        case 1
            if Y(i,:) ==[1 0 0]
                tp1=tp1+1;
            elseif Y(i,:)==[0 1 0]
                fp1=fp1+1;
                fn2=fn2+1;
            elseif Y(i,:)==[0 0 1]
                fp1=fp1+1;
                fn3=fn3+1;
            end
        case 2
            if Y(i,:) ==[1 0 0]
                fn1=fn1+1;
                fp2=fp2+1;
            elseif Y(i,:)==[0 1 0]
                tp2=tp2+1;
            elseif Y(i,:)==[0 0 1]
                fp2=fp2+1;
                fn3=fn3+1;
            end
        case 3
            if Y(i,:) ==[1 0 0]
                fp3=fp3+1;
                fn1=fn1+1;
            elseif Y(i,:)==[0 1 0]
                fp3=fp3+1;
                fn2=fn2+1;
            elseif Y(i,:)==[0 0 1]
                tp3=tp3+1;
                
            end
     end        
end
Cost=(norm(Y'-h_theta)^2)/(2*m);
R1=tp1/(tp1+fn1);
R2=tp2/(tp2+fn2);
R3=tp3/(tp3+fn3);
P1=tp1/(tp1+fp1);
P2=tp2/(tp2+fp2);
P3=tp3/(tp3+fp3);
F1=2*(P1*R1)/(P1+R1);
F2=2*(P2*R2)/(P2+R2);
F3=2*(P3*R3)/(P3+R3);
F1=(F1+F2+F3)/3;



accuracy = mean(mean(double(Y_test == Y ))) * 100;
fprintf('\nSet Accuracy: %f\n', accuracy);
fprintf('\nSet F1 score: %f\n', F1);
fprintf('\nSet Cost: %f\n', Cost);
end

