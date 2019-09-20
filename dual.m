function [w,b,k]=dual(X,Y) %function for dual perceptron
alpha=ones(1,numel(X)/l); %first iteration of while loop will trigger
b=0; %bias initalised at zero.
l=length(Y); %number of elements to be separated is set
while alpha>0
    alpha=zeros(1,numel(X)/l); %iteration loop reset so will only trigger with zero errors
    R=0; %iterative counter for maximum value of x set
    Rmax=0; %maximum value of norm initialised
    for i=1:l
        R=dot(X(:,i),X(:,i)); %calculate norm of X
        if R>Rmax
            Rmax=R; %iterate until maximum of vector found
        end
    end
    for i=1:l
        g=0; %initalise functional margin
        for j=1:l
            g=g+a(j)*Y(j)*dot(X(:,j),X(:,i))+b; %decision function
        end
        if Y(i)*g<=0 %if functional margin is negative
            alpha(i)=alpha(i)+1; %update counter
            b=b+(Y(i)*Rmax*Rmax); %update bias
        end
    end
end
end