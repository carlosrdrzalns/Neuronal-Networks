function [h_theta] = FuncTheta(w2,w3,w4,b2,b3,b4,X)
% given NN params calculates h_tehta
    a1=X';
    z2=w2*a1+b2; %first we calculate the weighted input
    a2=sigmoid(z2); %apply the sigmoid function to each Z
    z3=w3*a2+b3;
    a3=sigmoid(z3);
    z4=w4*a3+b4;
    a4=sigmoid(z4);
    h_theta=a4;
end

