function CRP_alg(data, cc)

figure(2),

[N,D]=size(data);

% stardarize y by making each column be zero mean, standard derivation 1.
y = data - repmat(mean(data), [N 1]);
y = y ./ repmat(std(data), [N 1]);

numdata=size(y,1);


set(gcf,'units','points','position',[100,100,1000,300])

subplot(121)
gscatter(y(:,1), y(:,2), cc)
title('Ground Truth')
box on
drawnow;

%% Neal's Algorithm 8

% initial cluster number
initcluster=1;
initcc=rem(randperm(numdata),initcluster)+1; % randomly set the data points to initial cluster

for i=1:initcluster
    [indx]=find(initcc(:)==i);
    classindx{i}=indx;
end

numclass=length(classindx);

for i=1:numclass
    ind=classindx{i};
    for j=1:length(ind)
        dataclass(ind(j))=i;
    end
end

% plot the initial clusters
subplot(122)

gscatter(y(:,1), y(:,2), initcc)
hold on

clear dataclassmean;

% plot the ellipse for initial cluster
for ci=1:numclass
    ind=classindx{ci};
    sub_y=[];
    for n=1:length(ind)
        sub_y(n,:)=y(ind(n),:);
    end
    mu = mean(sub_y,1 );
    tmp = bsxfun(@minus,sub_y, mu);
    covar=cov(tmp);
    eclipse_plot(mu, covar);
end

box on
title('Initial' ,'FontSize',15);
drawnow;

alpha=10;
iter=0;
while iter<100 % number of iterations
    iter=iter+1;
    spirit=0;
    
    for nn=randperm(numdata)%1:numdata  % scan all the data points
        % prior G0:
        m=5;
        G0means=mean(y,1); 
        for g=1:m
            ghost(g,1)=G0means(1,1)+randn*10;
            ghost(g,2)=G0means(1,2)+randn*10;
            ghostsigma(:,:,g)=wishrnd([1 0;0 1], 10);%+.0001 * eye(2);
        end
        
        %scanned data point
        data=y(nn,:);
        clear p;
        
        sigma=[1 0;0 1]; % for simple experiment, sigma is set to 1 constant
        for ci=1:numclass+spirit
            
            % calculate the probability for each existing cluster
            ind=classindx{ci};
            sub_y=[0 0];
            for n=1:length(ind)
                if ind(n)~=nn  % remove the scanning data point n_(-i)
                    sub_y(n,:)=y(ind(n),:);
                end
            end
            
            mucc=mean(sub_y,1);
            
            muhat=repmat(mucc,size(sub_y,1),1);
            sigma=(sub_y-muhat)'*(sub_y-muhat)./(size(sub_y,1)-1)+.0001 * eye(2);
            if sum(sum(sigma))==0
                sigma=[1 0;0 1];
            elseif isnan(sigma)
                sigma=[1 0;0 1];
            end
            
            if ~isempty(mucc)
                n=length(sub_y);
                mucc2(1,1)=(n/(1+n))*mucc(1,1)+(1/(n+1))*G0means(1,1);
                mucc2(1,2)=(n/(1+n))*mucc(1,2)+(1/(n+1))*G0means(1,2);
                % Case 1: 1<c<k-
                % p(c_i|c_(-i),theta)=[n_(-i,c)/(n-1+alpha)]*F(y_i|theta_c)
                p1=mvnpdf(data,mucc2,sigma);
                p2=(length(sub_y))/(numdata-1+alpha);
                p(ci)=p1*p2;
            end
        end
        
        for ci=(numclass+spirit+1):(numclass+spirit+m)
            % Case 2: k-<c<h
            % p(c_i|c_(-i),theta)=[alpha/(n-1+alpha)]*F(y_i|theta_c)
            p1=mvnpdf(data,ghost(ci-numclass-spirit,:),ghostsigma(:,:,ci-numclass-spirit));
            p2=(alpha/m)/(numdata-1+alpha);
            p(ci)=p1*p2;
        end
        
        clear temp;
        temp=p;
        [tr,tl]=size(temp);
        
        temp=temp/norm(temp);
        
        % Gibbs Mehod:
        temp = cumsum(temp,2);
        temp=temp./temp(tr,tl);
        thr=rand;
        tn=1;
        for tj=1:tl
            if temp(tj)<thr*temp(tr,tl)
                tn=tn+1;
            end
        end
        
        % add removed data back to some cluster
        dataclass(nn)=tn;
        
        if tn>numclass+spirit
            % ghost became spirit, which formed the new cluster and kept for the next scanning points
            spirit=spirit+1;
            dataclass(nn)=numclass+spirit;
            classindx{numclass+spirit}=numclass+spirit;
            y(numdata+spirit,1)=ghost(tn-numclass-(spirit-1),1);
            y(numdata+spirit,2)=ghost(tn-numclass-(spirit-1),2);
        end
        
    end  % finish scanning all the data points
    
    clear classindx
    ci=0;
    for i=1:(numclass+spirit)
        [indx]=find(dataclass(:)==i);
        if(~isempty(indx))  % remove empty class
            ci=ci+1;
            classindx{ci}=indx;
        end
    end
    
    numclass=length(classindx);
    % relabel
    for i=1:numclass
        ind=classindx{i};
        for j=1:length(ind)
            dataclass(ind(j))=i;
        end
    end
    
    % plot the new clusters after every iteration
    figure(2),
    subplot(122)
    hold off
    for ci=1:numclass
        ind=classindx{ci};
        sub_y=[];
        for n=1:length(ind)
            sub_y(n,:)=y(ind(n),:);
        end
        gscatter(y(1:numdata,1), y(1:numdata,2), dataclass)
        hold on
        mu = mean(sub_y,1 );
        tmp = bsxfun(@minus,sub_y, mu);
        covar=cov(tmp);
        eclipse_plot(mu, covar);
        box on
        title(strcat('CRP Iteration#', num2str(iter)),'FontSize',15)
    end
    drawnow;
    
end

