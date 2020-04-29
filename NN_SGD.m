%inicializamos las variables a un valor aleatorio:
input_layer=size(X_train,2);
hidden_layer_size=1250;
num_labels=size(Y_train,2);
[w2,b2]=randInitializeWeights(input_layer, hidden_layer_size);
[w3,b3]=randInitializeWeights(hidden_layer_size, hidden_layer_size);
[w4,b4]=randInitializeWeights(hidden_layer_size, num_labels);
lambda=0;
learningRate=0.1;
Nepoch=5;
minibatchsize=50;
evalIter=200;
m=size(X_train,1);
% Cost=zeros(20950/50,Nepoch);


%Realizamos el algoritmo de BP con la función computeGrad y el learning
%rate asociado:
s='cross-entropy';
v=zeros(round(m/evalIter,0),3);
for i=1:Nepoch
    fprintf('epoch number : %d \n',i);
    for j=1:minibatchsize:20950
        n=j+49;
        %Hacemos el calculo de las derivadas parciales
        [Cost, Db2, Db3, Db4, Dw2, Dw3, Dw4]=ComputeGrad(w2,w3,w4,b2,b3,b4,X_train(j:n,:),Y_train(j:n,:),s,0);
        fprintf('iteration number: %d       Cost: %d \n',n/50, Cost);
        %Actualizamos los parámetros de la red
        w2=w2*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw2;
        w3=w3*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw3;
        w4=w4*(1-learningRate*lambda/minibatchsize)-(learningRate/minibatchsize)*Dw4;
        b2=b2-(learningRate/minibatchsize)*Db2;
        b3=b3-(learningRate/minibatchsize)*Db3;
        b4=b4-(learningRate/minibatchsize)*Db4;
%         if mod(n/evalIter,1)==0
%             [J_val, F1, Accu]=TestNN(w2,w3,w4,b2,b3,b4,X_val, Y_val);
%             v(n/evalIter, :)=[J_val, F1, Accu];
%             subplot(3,1,1)
%             title('Cost function validation')
%             xlabel('iteration')
%             ylabel('Cost')
%             plot(n/evalIter, J_val, '-b');
%             subplot(3,1,2)
%             title('F1 score validation')
%             xlabel('iteration')
%             ylabel('F1')
%             plot(n/evalIter, F1, '-r');
%             title('Accuracy validation')
%             xlabel('iteration')
%             ylabel('Accuracy')
%             plot(n/evalIter, Accu, '-g');
%         end   
    end
%     [X_train,Y_train]=ShuffleMatrix(X_train, Y_train);
end
fprintf('Fin del entrenamiento')
[J_val, F1, Accu]=TestNN(w2,w3,w4,b2,b3,b4,X_val, Y_val);
fprintf('valor de Coste final:        %d \n', J_val);
fprintf('valor de F1 score final:     %d \n', F1);
fprintf('valor de Accuracy final:     %d \n', Accu);
