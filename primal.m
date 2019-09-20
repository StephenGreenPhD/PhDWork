function [w,b,k]=primal(X,Y,eta) %function for primal perceptron
k=1; %first iteration of while loop will trigger
b=0; %bias initalised at zero.
l=length(Y); %number of elements to be separated is set
w=zeros(1,numel(X)/l); %weight vector initalised at zero.
while k>0
    k=0; %iteration loop reset so will only trigger with zero errors
    R=0; %iterative counter for maximum value of x set
    Rmax=0; %maximum value of norm initialised
    for i=1:l
        R=dot(X(:,i),X(:,i)); %calculate norm of X
        if R>Rmax
            Rmax=R; %iterate until maximum of vector found
        end
    end
    for i=1:l
        if Y(i)*(dot(w,X(:,i))+b)<=0 %if functional margin is negative
            w=w+(eta*Y(i)*X(:,i))'; %update weight
            b=b+eta*Y(i)*Rmax; %update bias
            k=k+1; %update counter
        end
    end
end
end