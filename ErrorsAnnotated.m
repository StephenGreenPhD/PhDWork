load Bottlenecks; %Bottlenecks L=26000x2048 and truths N=26000x1 retrived.
Ndata=size(L,1); %Note number of Bottlenecks
L=L-repmat(mean(L),Ndata,1);% Center data
[H,s,l]=pca(L);% Kaiser dimensionality estimate
Dim=sum(l>mean(l));% Dimensionality reduction
Xr=L*H(:,1:Dim);% Capture dimensions that pass Kaiser rule
[Hr, sr, lr] = pca(Xr); %Reorganise data
Xw=Xr*Hr*diag(1./sqrt(lr));% Whiten data
for i=1:Ndata
    Xw(i,:)=Xw(i,:)/norm(Xw(i,:)); % Normalise Data
end% Testing how far are we from a uniformly covered sphere
Sphere_distortion=norm(mean(Xw));
sprintf('Spherical distortion = %d', Sphere_distortion)
% Forming the set of correct responses as well as the set of errors.
Xcorrect=zeros(sum(N==1),Dim);% Correct response counter
Xerrors=zeros(sum(N==0),Dim);% Incorrect response counter
Cr_counter=0;% Correct response counter
Er_counter=0;% Error response counter
for i=1:Ndata
    if (N(i)==1)
        Cr_counter=Cr_counter+1; %If correct in ground truth add to tally
        Xcorrect(Cr_counter,:)=Xw(i,:);
    else
        Er_counter=Er_counter+1; %If incorrect in ground truth add to tally
        Xerrors(Er_counter,:)=Xw(i,:);
    end
end
% Testing: the sum of two counters must be equal to Ndata
sprintf('Total: %u, Labelled as Correct: %u, Lebelled as Errors: %u ',...
    Ndata, Cr_counter, Er_counter)
% Show any clustering structure
Num_trials=12; %Determine Number Of Trials
ErrorNumber=zeros(1,Er_counter);
for i=1:Er_counter
    ErrorNumber(i)=i; %Locate position of all errors
end
CollectedTP=zeros(10*Num_trials,Er_counter); %Log all trial values
AverageTP=zeros(10,Er_counter); %Log all average of trial values
for t=1:10 %Determine each hyperplane threshold
    FinalTP=[]; %Collect all successful error hyperplanes 
    parfor k=1:Num_trials
        TP=ones(1,Er_counter); %Log all True Positive Rates
        A=0; %Set while loop
        j=0; %Number of times loop has iterated
        while A~=1
            j=j+1; %Increase for each iteration
            Num_clusters=j*10; %Increase clusters in multiples of 10
            [idx, Centroids]=kmeans(Xerrors,Num_clusters); %perform k means
            [buff, cluster_pointer]=sort(idx); %Rearrange errors by cluster
            Xerrors_sorted_by_cluster=zeros(sum(N==0),Dim);
            for i=1:Er_counter
                Xerrors_sorted_by_cluster(i,:)=...
                    Xerrors(cluster_pointer(i),:);
            end
            %Deteriming the number of points in each cluster 
            %For this we first introduce cluster swiping counter, 
            %csc, and size_cluster variables 
            %We will also use sorted indexes of pointers to clusters,
            %stored in buff array (derived previously)
            csc=1; %starts counting first cluster
            size_cluster=zeros(1,Num_clusters); %No. of elements in cluster
            for i=1:Er_counter
                if (buff(i)==csc) %Add to running tally is matched
                    size_cluster(csc)=size_cluster(csc)+1;
                else
                    csc=csc+1; %Move to next cluster if mismatched
                    size_cluster(csc)=size_cluster(csc)+1;
                end
            end
            % Determining corrector weights and corresponding thresholds 
            % Pick clusters' centroids as the corrector weights 
            % Covariance matrices are likely diagonally-dominated
            w=zeros(Num_clusters,Dim); %Prepare tally of hyperplanes
            NewC=zeros(1,0); % Tally of successful hyperplanes
            NewW=zeros(0,Dim); % Error tally that meets hyperplane criteria 
            bounds=cumsum([0 size_cluster]); %calculate stopping points
            X=(t-1)*0.1; %adjustable hyperplane threshold (0.0 to 0.9)
            for i=1:Num_clusters %create hyperplane
                w(i,:)=Centroids(i,:)*inv(cov(Xcorrect)+...
                    cov(Xerrors(1+bounds(i):bounds(i+1),:)));
                w(i,:)=w(i,:)/norm(w(i,:)); %normalise
                %If threshold met then record errors and positions
                if min(Xerrors_sorted_by_cluster...
                        (1+bounds(i):bounds(i+1),:)*w(i,:)'-0.00001)>=X
                    NewC=[NewC min(Xerrors_sorted_by_cluster...
                        (1+bounds(i):bounds(i+1),:)*w(i,:)'-0.00001)];
                    NewW=[NewW; w(i,:)];
                end
            end
            %determine second hyperplane
            ZError=(sign(Xerrors*NewW'-repmat(NewC,Er_counter,1))+1)/2;
            %Add up all component dimensions
            Er_cumulative=sum(ZError,2);
            %Record all hyperplanes that have flagged at least one error
            %Any non-zero number in Cr_response indicates a False Positive        
            %Summing results in the number of false positives in the system
            Er_response=sum(Er_cumulative>0);        
            %Calculate percentage of flagged errors of total errors
            A=Er_response/Er_counter;
            [t k j Num_clusters]
            TP(1,j)=A; %Record percentage of flagged errors
        end
        FinalTP=[FinalTP; TP]; %Add to running tally for each trial
    end
    %Collect for all hyperplane thresholds and number of trials
    CollectedTP(((t-1)*Num_trials)+1:t*Num_trials,:)=FinalTP;
    %Calculate averages for further analysis
    AverageTP(t,:)=mean(FinalTP);
end
%Plot results of number of clusters over success rate using boxplots.
figure(2);
boxplot(FinalTP',ErrorNumber, 'LabelOrientation', 'inline');