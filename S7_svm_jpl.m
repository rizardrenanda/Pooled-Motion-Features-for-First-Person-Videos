%% DOCUMENTATION


% OBJECTIVES
% We followed the standard evaluation
% setting of the DogCentric dataset: we performed repeated
% random training/testing splits 100 times, and averaged the
% performance. We randomly selected half of videos per activity
% class as training videos, and used the others for the
% testing. If the number of total videos per class is odd, we
% add one more video

% DEPENDENCIES
% 1. LIBSVM plus chiSquare
% File already stored in Dependencies Folder. Just add thats folder
% to your MATLAB system path

% EVALUATION SETTING
% Input     : features representation 
% Output    : 100 separated features for training and testing. 

%%
clc
clear

% load data
disp('import data');

load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\label_jpl1
load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\label_jpl2
load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\hht_1_5\spatial_s51_hht_1_5
load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\pot_cafee_jpl
% load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\fv_tdd_jpl
% load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\fv_handcrafted
load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\fv_idt_jpl

% do normalization for each descriptor
disp('do normalization');
norm_type = 'L2';

% IDT
data7 = func_order_jpl(func_norm(double(fv_mbh),norm_type));
data8 = func_order_jpl(func_norm(double(fv_tra),norm_type));
data9 = func_order_jpl(func_norm(double(fv_hog),norm_type));
data10 = func_order_jpl(func_norm(double(fv_hof),norm_type));

% TDD
% data13 = func_norm(double(fv_s51),norm_type);
% data14 = func_norm(double(fv_s52),norm_type);
% data15 = func_norm(double(fv_t41),norm_type);
% data16 = func_norm(double(fv_t42),norm_type);

% POT
data_tmp11 = double(caffe_pot_sum)';
data11 = func_order_jpl(func_norm(data_tmp11, norm_type));
data_tmp12 = caffe_pot_grad2';
data12 = func_order_jpl(func_norm(data_tmp12, norm_type));

% % HHT
data171 = func_order_jpl(hht_final_mean);
data181 = func_order_jpl(hht_final_std);
data191 = func_order_jpl(hht_final_centroid_spectral);
data201 = func_order_jpl(hht_final_varian_coeff);
data211 = func_order_jpl(hht_final_entropy);
data221 = func_order_jpl(hht_final_entropy_instan);
data231 = func_order_jpl(hht_final_mean_instan);
data241 = func_order_jpl(hht_final_mean_energy);
data251 = func_order_jpl(hht_final_entropy_energy);
data261 = func_order_jpl(hht_final_std_energy);

load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\hht_1_5\spatial_s52_hht_1_5
data172 = func_order_jpl(hht_final_mean);
data182 = func_order_jpl(hht_final_std);
data192 = func_order_jpl(hht_final_centroid_spectral);
data202 = func_order_jpl(hht_final_varian_coeff);
data212 = func_order_jpl(hht_final_entropy);
data222 = func_order_jpl(hht_final_entropy_instan);
data232 = func_order_jpl(hht_final_mean_instan);
data242 = func_order_jpl(hht_final_mean_energy);
data252 = func_order_jpl(hht_final_entropy_energy);
data262 = func_order_jpl(hht_final_std_energy);

load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\hht_1_5\temporal_t41_hht_1_5
data173 = func_order_jpl(hht_final_mean);
data183 = func_order_jpl(hht_final_std);
data193 = func_order_jpl(hht_final_centroid_spectral);
data203 = func_order_jpl(hht_final_varian_coeff);
data213 = func_order_jpl(hht_final_entropy);
data223 = func_order_jpl(hht_final_entropy_instan);
data233 = func_order_jpl(hht_final_mean_instan);
data243 = func_order_jpl(hht_final_mean_energy);
data253 = func_order_jpl(hht_final_entropy_energy);
data263 = func_order_jpl(hht_final_std_energy);


load C:\Users\didpurwanto\Documents\Dataset\JPL\feature_representatin\hht_1_5\temporal_t42_hht_1_5
data174 = func_order_jpl(hht_final_mean);
data184 = func_order_jpl(hht_final_std);
data194 = func_order_jpl(hht_final_centroid_spectral);
data204 = func_order_jpl(hht_final_varian_coeff);
data214 = func_order_jpl(hht_final_entropy);
data224 = func_order_jpl(hht_final_entropy_instan);
data234 = func_order_jpl(hht_final_mean_instan);
data244 = func_order_jpl(hht_final_mean_energy);
data254 = func_order_jpl(hht_final_entropy_energy);
data264 = func_order_jpl(hht_final_std_energy);

disp('combine data');
% choose data
data_itf = [data8 data9 data7];
data_pot = [data11 data12];

% data_hht = [entropy_energy                spectral_centroid_freq          mean_instan                     mean_analytic]
% data_hht = [data251 data252 data253 data254 data191 data192 data193 data194 data231 data232 data233 data234 data171 data172 data173 data174];
% data_norm = func_norm(data_hht,norm_type);

data_hht1 = [data251 data252 data253 data254];
data_hht2 = [data191 data192 data193 data194];
data_hht3 = [data231 data232 data233 data234];
data_hht4 = [data171 data172 data173 data174];


data_norm1 = func_norm(data_hht1,norm_type);
data_norm2 = func_norm(data_hht2,norm_type);
data_norm3 = func_norm(data_hht3,norm_type);
data_norm4 = func_norm(data_hht4,norm_type);

% data_darwin = [data10b_norm data10f_norm];
data = [data_pot data_itf data_norm1 data_norm2 data_norm3 data_norm4];

% data_darwin = [data10b_norm data10f_norm];
% data = [data_itf];


[x,y] = size(data);

% randoming
label = label';
training_label = label2';

cls1 = 1:12;
cls2 = 13:24;
cls3 = 25:36;
cls4 = 37:48;
cls5 = 49:60;
cls6 = 61:72;
cls7 = 73:84;
ori = 1:84;

training_set = [];
testing_set = [];
result = [];
index_train = [];
index_test = [];

disp('do randoming video');
for i = 1:100
    % random for training and testing
    index1 = randperm(numel(cls1));
    % nums1 = 1:length(cls1)/2;
    c1 = cls1(index1);
    train1 = c1(1:length(cls1)/2);
    test1 = c1((length(cls1)/2+1):length(cls1));
    
    index2 = randperm(numel(cls2));
    nums2 = 1:length(cls2)/2;
    c2 = cls2(index2);
    train2 = c2(1:length(cls2)/2);
    test2 = c2(length(cls2)/2+1:length(cls2));
    
    index3 = randperm(numel(cls3));
    nums3 = 1:length(cls3)/2;
    c3 = cls3(index3);
    train3 = c3(1:length(cls3)/2);
    test3 = c3(length(cls3)/2+1:length(cls3));
    
    index4 = randperm(numel(cls4));
    nums4 = 1:length(cls4)/2;
    c4 = cls4(index4);
    train4 = c4(1:length(cls4)/2);
    test4 = c4(length(cls4)/2+1:length(cls4));
    
    index5 = randperm(numel(cls5));
    nums5 = 1:length(cls5)/2;
    c5 = cls5(index5);
    train5 = c5(1:length(cls5)/2);
    test5 = c5(length(cls5)/2+1:length(cls5));
    
    index6 = randperm(numel(cls6));
    nums6 = 1:length(cls6)/2;
    c6 = cls6(index6);
    train6 = c6(1:length(cls6)/2);
    test6 = c6(length(cls6)/2+1:length(cls6));
    
    index7 = randperm(numel(cls7));
    nums7 = 1:length(cls7)/2;
    c7 = cls7(index7);
    train7 = c7(1:length(cls7)/2);
    test7 = c7(length(cls7)/2+1:length(cls7));
         
    % index training and testing
    index_training = [train1 train2 train3 train4 train5 train6 train7];
    index_testing = [test1 test2 test3 test4 test5 test6 test7];
    
    index_train = vertcat(index_train,index_training);
    index_test = vertcat(index_test,index_testing);
    
    for j = 1 : 42
        training_set(j,:) = data(index_training(j),:);
        testing_set(j,:) = data(index_testing(j),:);
    end
    
    
    % SVM training  
    aa = sprintf('do training ...........................................(%d/%d)',i,100);
    disp(aa);
    libsvm_options = '-s 0 -t 0 -c 100';
    
    % SVM testing
    model = svmtrain(training_label, training_set, libsvm_options);
    [predicted_label, accuracy, decision_values_prob_estimates] = svmpredict(training_label, testing_set, model);
    
    predicted_label_result(:,i) = predicted_label;

    % reset
    training_set = [];
    testing_set = [];
    result(i) = accuracy(1);  
    disp(result);
    disp('mean average:');
    disp(mean(result));
end

average_result = mean(result);
disp(average_result);

save('result.mat','result','predicted_label_result','index_train','index_test');