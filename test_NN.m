
[h_theta]=FuncTheta(Theta1,Theta2,Theta3,X);% Funcion que predice el valor 3 dimensiones en la que el valor mayor de probabilidad indica el objeto que corresponde
for i=1:size(y,1)
    [M,I]=max(h_theta(:,i));
    Y_test(i)=I;
end

accuracy = mean(double(Y_test' == y)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', accuracy);

    

