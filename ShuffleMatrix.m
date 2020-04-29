function [Xout,Yout] = ShuffleMatrix(Xin,Yin)
%randomly shuffle matrix X & vector Y
m=size(Xin,1);
v=randperm(m);
    for i=1:m
        Xin(i,:)=Xin(v(i),:);
        Yin(i,:)=Yin(v(i),:);
    end
Xout = Xin;
Yout = Xout;
end

