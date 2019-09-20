load('Bottlenecks.mat') %Bottleneck matrix L=26000x2048 retrived
%[Zica,W,T,mu]=fastICA(L',2048); %Comment for non FastICA variant
%L=Zica'; %Comment for non FastICA variant
L=L-repmat(mean(L),size(L,1),1); %Centering process
[Dr,s,l]=pca(L); %PCA process for bottlenecks
Xr=L*Dr(:,1:sum(l>mean(l))); %Regularisation process (Kaiser Rule)
[Hr, sr, lr] = pca(Xr); %Re-regularise for component shift
Xw=Xr*Hr*diag(1./sqrt(lr)); %Whiten result
[X,Y]=size(Xw); %Retrive number of images and parameters
Phi=zeros(2,Y); %Save largest values with and without kernel trick
New=zeros(Y*(Y+1)/2+Y,X); %Set up kernel trick
NewNinPhi=Xw2; %Set table of largest kernel captures, keep structure
for k=1:Y %Work backwards to save calculation time
    Xw2=Xw(:,1:k); %Retrive relevant components
    Xw3=(Xw2-repmat(mean(Xw2,2),1,Y))';% Centering of the kernelled data
    parfor j=1:Y % Translation of the matrix for evaluation of distances
        NewNinPhi(j,k)=sum(sqrt(sum((Xw3-repmat(Xw3(j,:)/2,X,1)).^2,2))...
            <=norm(Xw3(j,:)/2)); %For largest value Y
    end
    Phi(1,k)=max(NewNinPhi(:,k));%Record the most values captured in kernel
    save('KernelExampleICA.mat','Phi') %Save all non-trick results
end
parfor j=1:X %Prepare kernel trick matrix
    Square=Xw(:,j)*Xw(:,j)'; %Retrive product of every combination
    Square=triu(Square); %Remove duplicate values
    New(:,j)=cat(1,flip(Xw(:,j)),nonzeros(Square)); %Concatenate
end
for k=1:Y
    New2=New(E-k+1:E+((k*(k+1))/2),:); %Retrieve relevant components
    New3=(New2-repmat(mean(New2,2),1,Y))';% Centering of the kernelled data
    parfor j=1:Y % Translation of the matrix for evaluation of distances
        NewNinPhi(j,k)=sum(sqrt(sum((New3-repmat(New3(j,:)/2,X,1)).^2,2))...
            <=norm(New3(j,:)/2)); %For largest value Y
    end
    Phi(2,k)=max(NewNinPhi(:,k)); %Record most values captured in kernel
    save('KernelExampleICA.mat','Phi') %Save all trick results
end