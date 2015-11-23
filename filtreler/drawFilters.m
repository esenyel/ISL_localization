close all
clear all
clc

numbers = [0,6,12,18,36];



for i = 1:length(numbers)
 
    fileName = sprintf('filtre%d.txt',numbers(i));
 
    sss = textread(fileName);   
    
    sss2 = reshape(sss,29,29);
    
    subplot(1,5,i); imshow(sss2);
end

