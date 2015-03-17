load('parkinsonsTestStatML.dt');
load('parkinsonsTrainStatML.dt');

% 2.1




traindata = parkinsonsTrainStatML(:,1:22);
traintarget = parkinsonsTrainStatML(:,23);
testdata = parkinsonsTestStatML(:,1:22);
testtarget = parkinsonsTestStatML(:,23);



normtrain = zeros(size(traindata));
normtest = zeros(size(testdata));

for i=1:size(traindata,2),
    normtrain(:,i) = normcol(traindata(:,i),traindata(:,i));
    normtest(:,i) = normcol(testdata(:,i),traindata(:,i));
end


% 2.2

C = [10, 100, 1000, 10000, 100000, 1000000, 10000000];
Gamma = [2^-21,2^-20,2^-19,2^-18,2^0, 2^5, 2^20];

errors = zeros(7,7);

[t1,t2,t3,t4,t5] = splitdata(traindata,traintarget);

sel1 = [t2 ; t3 ; t4 ; t5];
sel2 = [t1 ; t3 ; t4 ; t5];
sel3 = [t1 ; t2 ; t4 ; t5];
sel4 = [t1 ; t2 ; t3 ; t5];
sel5 = [t1 ; t2 ; t3 ; t4];


for k=1:5
    switch k
        case 1
            test = t1;
            train = sel1;
        case 2
            test = t2;
            train = sel2;
        case 3
            test = t3;
            train = sel3;
        case 4
            test = t4;
            train = sel4;
        case 5
            test = t5;
            train = sel5;
    end
    
    for i=1:7
        for j=1:7

            modelt = svmtrain(train(:,23),train(:,1:22), sprintf('-t 3 -c %f -g %f', C(i), Gamma(j)));
            [tresult,~,~] = svmpredict(test(:,23),test(:,1:22),modelt);
            errors(i,j) = errors(i,j) + sum(tresult + test(:,23) == 1);

        end
    end
end

[bC, bGamma] = find(errors==min(min(errors)),1);

bestGamma = Gamma(bGamma);
bestC = C(bC);

model = svmtrain(traintarget,traindata, sprintf('-t 3 -c %f -g %f', bestC, bestGamma));
[tresult,~,~] = svmpredict(traintarget,traindata,model);
traincrossvalerror = sum(tresult + traintarget == 1) / length(traindata);

[tresult,~,~] = svmpredict(testtarget,testdata,model);
testcrossvalerror = sum(tresult + testtarget == 1) / length(testdata);

%Normalized



C = [2^-1,2^2,2^3,2^4,2^6,2^8,2^10];
Gamma = [2^-13,2^-10,2^-6,2^-3,2,2^3,2^6];

errorsn = zeros(7,7);


[t1,t2,t3,t4,t5] = splitdata(normtrain,traintarget);

sel1 = [t2 ; t3 ; t4 ; t5];
sel2 = [t1 ; t3 ; t4 ; t5];
sel3 = [t1 ; t2 ; t4 ; t5];
sel4 = [t1 ; t2 ; t3 ; t5];
sel5 = [t1 ; t2 ; t3 ; t4];


for k=1:5
    switch k
        case 1
            test = t1;
            train = sel1;
        case 2
            test = t2;
            train = sel2;
        case 3
            test = t3;
            train = sel3;
        case 4
            test = t4;
            train = sel4;
        case 5
            test = t5;
            train = sel5;
    end
    
    for i=1:7
        for j=1:7

            modelt = svmtrain(train(:,23),train(:,1:22), sprintf('-t 3 -c %f -g %f', C(i), Gamma(j)));
            [tresult,~,~] = svmpredict(test(:,23),test(:,1:22),modelt);
            errorsn(i,j) = errorsn(i,j) + sum(tresult + test(:,23) == 1);

        end
    end
end

[bC, bGamma] = find(errorsn==min(min(errorsn)),1);

gammaNorm = Gamma(bGamma);
cNorm = C(bC);


modeln = svmtrain(traintarget,normtrain, sprintf('-t 3 -c %f -g %f', cNorm, gammaNorm));
[tresult,~,~] = svmpredict(traintarget,normtrain,modeln);
ntraincrossvalerror = sum(tresult + traintarget == 1) / length(traindata);

[tresult,~,~] = svmpredict(testtarget,normtest,modeln);
ntestcrossvalerror = sum(tresult + testtarget == 1) / length(testdata);


